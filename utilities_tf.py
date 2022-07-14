import pickle
import gzip
import numpy as np

import tensorflow as tf


def load_batch_gcnn(sample_files):
    """
    Loads and concatenates a bunch of samples into one mini-batch.
    """
    c_features = []
    e_indices = []
    e_features = []
    v_features = []
    candss = []
    cand_choices = []
    cand_scoress = []

    # load samples
    for filename_tensor in sample_files:
        with gzip.open(str(filename_tensor.numpy(), 'utf-8'), 'rb') as f:
            sample = pickle.load(f)

        sample_state, _, sample_action, sample_cands, cand_scores = sample['data']

        sample_cands = np.array(sample_cands)
        cand_choice = np.where(sample_cands == sample_action)[0][0]  # action index relative to candidates

        c, e, v = sample_state
        c_features.append(c['values']) # c['values'].shape=(4196, 5)
        e_indices.append(e['indices']) # e['indices'].shape=(2, 32941)
        e_features.append(e['values']) # e['values'].shape=(32941, 1)
        v_features.append(v['values']) # v['values'].shape=(10100, 19)
        candss.append(sample_cands) # (23,)
        cand_choices.append(cand_choice) # 17
        cand_scoress.append(cand_scores) # list, len(...)=23

    n_cs_per_sample = [c.shape[0] for c in c_features]
    n_vs_per_sample = [v.shape[0] for v in v_features]
    n_cands_per_sample = [cds.shape[0] for cds in candss]

    # concatenate samples in one big graph
    c_features = np.concatenate(c_features, axis=0) # n x d
    v_features = np.concatenate(v_features, axis=0) # K x e
    e_features = np.concatenate(e_features, axis=0) # m x c
    
    # edge indices have to be adjusted accordingly
    # cv_shift 有两行，对应 e_indices 的两行，第一行对应 constraints，第二行对应 variables
    # 假设 n_cs_per_sample 为 [n1, n2, n3]，那么执行完下面这句之后，cv_shift[0,:]为[0,n1,n1+n2], 
    # 然后剩下的就是要把 cv_shift[0,:] 加到 e_indices 第一行的每一块中，也把cv_shift[1,:]的内容加到e_indices第二行的每一块中，
    # e_indices 横向每一块的长度分别是 K1, K2, K3...
    # 注意，list的加法就是 append
    cv_shift = np.cumsum([
            [0] + n_cs_per_sample[:-1], 
            [0] + n_vs_per_sample[:-1]  
        ], axis=1)
    
    # 用 j:(j+1) 而不是 j，这样可以保持维度不变
    # np.array的加法：shape为nxm的矩阵加上shape为nx1的矩阵，就是给第一个矩阵每一行都加上第二个矩阵的第二行的那个数
    e_indices = np.concatenate([e_ind + cv_shift[:, j:(j+1)] 
        for j, e_ind in enumerate(e_indices)], axis=1)
    # candidate indices as well
    candss = np.concatenate([cands + shift
        for cands, shift in zip(candss, cv_shift[1])]) # cands 指的就是变量的候选，所以加上shift的时候shift的值就是前面sample中变量的个数，即cv_shift[1]
    cand_choices = np.array(cand_choices) # choices 这里并没有加 shift，为什么？需要看看后面是不是处理了
    cand_scoress = np.concatenate(cand_scoress, axis=0)

    # convert to tensors
    c_features = tf.convert_to_tensor(c_features, dtype=tf.float32)
    e_indices = tf.convert_to_tensor(e_indices, dtype=tf.int32)
    e_features = tf.convert_to_tensor(e_features, dtype=tf.float32)
    v_features = tf.convert_to_tensor(v_features, dtype=tf.float32)
    n_cs_per_sample = tf.convert_to_tensor(n_cs_per_sample, dtype=tf.int32)
    n_vs_per_sample = tf.convert_to_tensor(n_vs_per_sample, dtype=tf.int32)
    candss = tf.convert_to_tensor(candss, dtype=tf.int32)
    cand_choices = tf.convert_to_tensor(cand_choices, dtype=tf.int32)
    cand_scoress = tf.convert_to_tensor(cand_scoress, dtype=tf.float32)
    n_cands_per_sample = tf.convert_to_tensor(n_cands_per_sample, dtype=tf.int32)

    return c_features, e_indices, e_features, v_features, n_cs_per_sample, n_vs_per_sample, n_cands_per_sample, candss, cand_choices, cand_scoress
