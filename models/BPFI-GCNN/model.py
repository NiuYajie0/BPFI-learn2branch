from networkx.drawing.nx_pylab import draw
import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import pickle
import dgl
import dgl.function as fn
import dgl.nn.tensorflow as dglnn
from dgl.nn.tensorflow.conv.graphconv import GraphConv
from keras import backend as b
import networkx as nx
import matplotlib.pyplot as plt
# import pygraphviz as pgv


class PreNormException(Exception):
    pass


class PreNormLayer(K.layers.Layer):      


    """
    Our pre-normalization layer, whose purpose is to normalize an input layer
    to zero mean and unit variance to speed-up and stabilize GCN training. The
    layer's parameters are aimed to be computed during the pre-training phase.
    """

    def __init__(self, n_units, shift=True, scale=True):
        # super(n_units, self).__init__()
        super().__init__()
        assert shift or scale

        if shift:
            self.shift = self.add_weight(
                name=f'{self.name}/shift',
                shape=(n_units,),
                trainable=False,
                initializer=tf.keras.initializers.constant(
                    value=np.zeros((n_units,)),
                ),
            )
        else:
            self.shift = None

        if scale:
            self.scale = self.add_weight(
                name=f'{self.name}/scale',
                shape=(n_units,),
                trainable=False,
                initializer=tf.keras.initializers.constant(
                    value=np.ones((n_units,)),
                ),
            )
        else:
            self.scale = None

        self.n_units = n_units
        self.waiting_updates = False
        self.received_updates = False

    def build(self, input_shapes):
        self.built = True
        
    def call(self, input):
        if self.waiting_updates:
            self.update_stats(input)
            self.received_updates = True
            raise PreNormException

        if self.shift is not None:
            input = input + self.shift

        if self.scale is not None:
            input = input * self.scale

        return input

    def start_updates(self):
        """
        Initializes the pre-training phase.
        """
        self.avg = 0
        self.var = 0
        self.m2 = 0
        self.count = 0
        self.waiting_updates = True
        self.received_updates = False

    def update_stats(self, input):
        """
        Online mean and variance estimation. See: Chan et al. (1979) Updating
        Formulae and a Pairwise Algorithm for Computing Sample Variances.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
        """
        assert self.n_units == 1 or input.shape[-1] == self.n_units, f"Expected input dimension of size {self.n_units}, got {input.shape[-1]}."

        input = tf.reshape(input, [-1, self.n_units])
        sample_avg = tf.reduce_mean(input, 0)
        sample_var = tf.reduce_mean((input - sample_avg) ** 2, axis=0)
        sample_count = tf.cast(tf.size(input=input) / self.n_units, tf.float32)

        delta = sample_avg - self.avg

        self.m2 = self.var * self.count + sample_var * sample_count + delta ** 2 * self.count * sample_count / (
                self.count + sample_count)

        self.count += sample_count
        self.avg += delta * sample_count / self.count
        self.var = self.m2 / self.count if self.count > 0 else 1

    def stop_updates(self):
        """
        Ends pre-training for that layer, and fixes the layers's parameters.        
        """
        assert self.count > 0
        if self.shift is not None:
            self.shift.assign(-self.avg)
        
        if self.scale is not None:
            self.var = tf.where(tf.equal(self.var, 0), tf.ones_like(self.var), self.var)  # NaN check trick
            self.scale.assign(1 / np.sqrt(self.var))
        
        del self.avg, self.var, self.m2, self.count
        self.waiting_updates = False
        self.trainable = False


class BaseModel(K.Model):
    """
    Our base model class, which implements basic save/restore and pre-training
    methods.
    """
    def pre_train_init(self):
        self.pre_train_init_rec(self, self.name)
        
    @staticmethod
    def pre_train_init_rec(model, name):
        for layer in model.layers:
            if isinstance(layer, K.Model):
                BaseModel.pre_train_init_rec(layer, f"{name}/{layer.name}")
            elif isinstance(layer, PreNormLayer):
                layer.start_updates()
    def pre_train_next(self):
        return self.pre_train_next_rec(self, self.name)
        
    @staticmethod
    def pre_train_next_rec(model, name):
        for layer in model.layers:
            if isinstance(layer, K.Model):
                result = BaseModel.pre_train_next_rec(layer, f"{name}/{layer.name}")
                if result is not None:
                    return result
            elif isinstance(layer, PreNormLayer) and layer.waiting_updates and layer.received_updates:
                layer.stop_updates()
                return layer, f"{name}/{layer.name}"
        return None
    def pre_train(self, *args, **kwargs):
        try:
            self.call(*args, **kwargs)
            return False
        except PreNormException:
            return True
    def save_state(self, path):
        with open(path, 'wb') as f:
            for v_name in self.variables_topological_order:
                v = [v for v in self.variables if v.name == v_name][0]
                pickle.dump(v.numpy(), f)
    def restore_state(self, path):
        with open(path, 'rb') as f:
            for v_name in self.variables_topological_order:
                v = [v for v in self.variables if v.name == v_name][0]
                v.assign(pickle.load(f))


class GCNPolicy(BaseModel):
    """
    Our bipartite Graph Convolutional neural Network (GCN) model.
    """

    def __init__(self):
        super().__init__()

        self.emb_size = 64
        self.cons_nfeats = 5
        self.edge_nfeats = 1
        self.var_nfeats = 19
        self.out_feats = 64
        self.activation = K.activations.relu
        self.initializer = K.initializers.Orthogonal()
        # self.initializer = K.initializers.glorot_uniform()
        # CONSTRAINT EMBEDDING
        self.cons_embedding = K.Sequential([
            PreNormLayer(n_units=self.cons_nfeats),
            K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
            K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
        ])

        # EDGE EMBEDDING
        self.edge_embedding = K.Sequential([
            PreNormLayer(self.edge_nfeats),
        ])

        # VARIABLE EMBEDDING
        self.var_embedding = K.Sequential([
            PreNormLayer(n_units=self.var_nfeats),
            K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
            K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
        ])

        self.feature_module_left = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=None, use_bias=True, kernel_initializer=self.initializer)
        ])

        self.feature_module_right = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=None, use_bias=False, kernel_initializer=self.initializer)
        ])
        
        self.heteroconv = dglnn.HeteroGraphConv({
            'edge1' : dglnn.GraphConv(in_feats = self.emb_size,out_feats = self.out_feats,norm="none",weight=False, bias=False),
            'edge2' : dglnn.GraphConv(in_feats = self.emb_size,out_feats = self.out_feats,norm="none",weight=False, bias=False)},
            aggregate='sum')


        self.post_conv_module = K.Sequential([
            PreNormLayer(1, shift=False),  # normalize after convolution
        ])  

        self.output_conv_module = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=None, kernel_initializer=self.initializer),
            K.layers.Activation(self.activation),
            K.layers.Dense(units=self.emb_size, activation=None, kernel_initializer=self.initializer),
        ])
        
        self.output_module = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
            K.layers.Dense(units=1, activation=None, kernel_initializer=self.initializer, use_bias=False),
        ])


        # build model right-away
        self.build([
            (None, self.cons_nfeats), 
            (2, None),  
            (None, self.edge_nfeats), 
            (None, self.var_nfeats),  
            # (None, ),
            (None, ), 
            (None, ), 
        ])

        # save / restore fix
        self.variables_topological_order = [v.name for v in self.variables]

        # save input signature for compilation
        self.input_signature = [
            (
                tf.TensorSpec(shape=[None, self.cons_nfeats], dtype=tf.float32),
                tf.TensorSpec(shape=[2, None], dtype=tf.int32),
                tf.TensorSpec(shape=[None, self.edge_nfeats], dtype=tf.float32),
                tf.TensorSpec(shape=[None, self.var_nfeats], dtype=tf.float32),
                tf.TensorSpec(shape=[None], dtype=tf.int32),
                tf.TensorSpec(shape=[None], dtype=tf.int32),
            ),
            tf.TensorSpec(shape=[], dtype=tf.bool),
        ]

    def build(self, input_shapes):
        c_shape, ei_shape, ev_shape, v_shape, nc_shape, nv_shape = input_shapes
        emb_shape = [None, self.emb_size]
        if not self.built:
            self.cons_embedding.build(c_shape)
            self.edge_embedding.build(ev_shape)
            self.var_embedding.build(v_shape)
            self.feature_module_left.build(emb_shape)
            self.feature_module_right.build(emb_shape) 
            self.heteroconv.build(emb_shape)
            self.post_conv_module.build(emb_shape)
            self.output_conv_module.build([None, self.emb_size + self.emb_size])
            self.output_module.build(emb_shape)
            # self.conv_v_to_c.build((emb_shape,emb_shape))
            # self.conv_c_to_v.build((emb_shape,emb_shape))
            # self.post_conv_module.build([None, self.emb_size])
            # self.output_module.build(emb_shape)
            self.built = True

    @staticmethod
    def pad_output(output, n_vars_per_sample, pad_value=-1e8):
        n_vars_max = tf.reduce_max(n_vars_per_sample)

        output = tf.split( # 在logits = model.pad_output(logits, n_cands.numpy())中n_vars_per_sample对应的其实是n_cands
            value=output,
            num_or_size_splits=n_vars_per_sample,
            axis=1,
        )
        output = tf.concat([
            tf.pad(
                x,
                paddings=[[0, 0], [0, n_vars_max - tf.shape(x)[1]]],
                mode='CONSTANT',
                constant_values=pad_value)
            for x in output
        ], axis=0)

        return output
    def call(self, inputs):
        """
        Accepts stacked mini-batches, i.e. several bipartite graphs aggregated
        as one. In that case the number of variables per samples has to be
        provided, and the output consists in a padded dense tensor.

        Parameters
        ----------
        inputs: list of tensors
            Model input as a bipartite graph. May be batched into a stacked graph.

        Inputs
        ------
        constraint_features: 2D float tensor
            Constraint node features (n_constraints x n_constraint_features)
        edge_indices: 2D int tensor
            Edge constraint and variable indices (2, n_edges)
        edge_features: 2D float tensor
            Edge features (n_edges, n_edge_features)
        variable_features: 2D float tensor
            Variable node features (n_variables, n_variable_features)
        n_cons_per_sample: 1D int tensor
            Number of constraints for each of the samples stacked in the batch.
        n_vars_per_sample: 1D int tensor
            Number of variables for each of the samples stacked in the batch.

        Other parameters
        ----------------
        training: boolean
            Training mode indicator
        """

        constraint_features, edge_indices, edge_features, variable_features, n_cons_per_sample, n_vars_per_sample = inputs
        a = variable_features.shape[0]
        b = constraint_features.shape[0]

         #facilities
        # variable_features1 = tf.zeros([a])
        # new = tf.Variable(variable_features)
        # new1 = tf.Variable(variable_features1)
        # new = new[:,4].assign(new1)

        # variable_features2 = tf.zeros([a,2])
        # new1 = tf.Variable(variable_features2)
        # new = new[:,7:9].assign(new1)


        # variable_features3 = tf.zeros([a,6])
        # new1 = tf.Variable(variable_features3)
        # new = new[:,10:16].assign(new1)

        # variable_features3 = tf.zeros([a,2])
        # new1 = tf.Variable(variable_features3)
        # new = new[:,17:19].assign(new1)

        # variable_features = tf.convert_to_tensor(new)


        # constraint_features1 = tf.zeros([b,2])
        # new = tf.Variable(constraint_features)
        # new1 = tf.Variable(constraint_features1)
        # new = new[:,0:2].assign(new1)

        # constraint_features2 = tf.zeros(b)
        # new1 = tf.Variable(constraint_features2)
        # new = new[:,4].assign(new1)

        # constraint_features = tf.convert_to_tensor(new)


        #indset
        
        # variable_features1 = tf.zeros([a,5])
        # new = tf.Variable(variable_features)
        # new1 = tf.Variable(variable_features1)
        # new = new[:,0:5].assign(new1)

        # variable_features2 = tf.zeros([a])
        # new1 = tf.Variable(variable_features2)
        # new = new[:,7].assign(new1)

        # variable_features2 = tf.zeros([a,6])
        # new1 = tf.Variable(variable_features2)
        # new = new[:,10:16].assign(new1)

        # variable_features = tf.convert_to_tensor(new)

        #cauctions


        #setcover:

        # variable_features1 = tf.zeros([a,4])
        # new = tf.Variable(variable_features)
        # new1 = tf.Variable(variable_features1)
        # new = new[:,0:4].assign(new1)

        # variable_features3 = tf.zeros([a])
        # new1 = tf.Variable(variable_features3)
        # new = new[:,17].assign(new1)

        # variable_features = tf.convert_to_tensor(new)



        n_cons_total = tf.math.reduce_sum(n_cons_per_sample) # numpy=55394  #numpy=49739
        n_vars_total = tf.math.reduce_sum(n_vars_per_sample) # numpy=80800  

        # EMBEDDINGS
        constraint_features = self.cons_embedding(constraint_features) #(97970, 64) #(49739, 64)
        edge_features = self.edge_embedding(edge_features) #(664478, 1)  #(339984, 1)
        variable_features = self.var_embedding(variable_features)  #=(161600, 64)  #(80800, 64)
        

        # GRAPH CONVOLUTIONS
        g = dgl.heterograph({
            ('variables', 'edge1', 'constraints'):(edge_indices[1],edge_indices[0]),
            ('constraints', 'edge2', 'variables'):(edge_indices[0],edge_indices[1])
         }) 
         
        h_src = {"variables":self.feature_module_right(variable_features)}
        h_dst = {"constraints":self.feature_module_left(constraint_features)}
        h1 = self.heteroconv(g,(h_src,h_dst))
        constraint_features1 = tf.reshape(h1["constraints"],(h1["constraints"].shape[0],-1))
        conv_output_c = self.post_conv_module(constraint_features1)
        constraint_features = self.output_conv_module(tf.concat([
            conv_output_c,
            constraint_features,
        ], axis=1))
        v_src = {"constraints":self.feature_module_left(constraint_features)}
        v1 = self.heteroconv(g,(v_src, h_src))
        variable_features1 = tf.reshape(v1["variables"],(v1["variables"].shape[0],-1))
        conv_output_v = self.post_conv_module(variable_features1)
        variable_features = self.output_conv_module(tf.concat([
            conv_output_v,
            variable_features,
        ], axis=1))

        # OUTPUT
        output = self.output_module(variable_features)
        output = tf.reshape(output, [1, -1]) # shape=(1, 161600)

        if n_vars_per_sample.shape[0] and n_vars_per_sample.shape[0] > 1:
            output = self.pad_output(output, n_vars_per_sample)

        return output


