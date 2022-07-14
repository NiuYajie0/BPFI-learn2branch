import os
import sys
import importlib
import argparse
import csv
import numpy as np
import time
import pickle
import pathlib
import gzip
import numpy

import tensorflow as tf
# import tensorflow.contrib.eager as tfe

# import svmrank

import utilities

from utilities_tf import load_batch_gcnn


def load_batch_flat(sample_files, feats_type, augment_feats, normalize_feats):
    cand_features = []
    cand_choices = []
    cand_scoress = []

    print(sample_files)

    for i, filename in enumerate(sample_files):
        cand_states, cand_scores, cand_choice = utilities.load_flat_samples(filename.numpy(), feats_type, 'scores', augment_feats, normalize_feats)

        cand_features.append(cand_states)
        cand_choices.append(cand_choice)
        cand_scoress.append(cand_scores)

    n_cands_per_sample = [v.shape[0] for v in cand_features]

    cand_features = np.concatenate(cand_features, axis=0).astype(np.float32, copy=False)
    cand_choices = np.asarray(cand_choices).astype(np.int32, copy=False)
    cand_scoress = np.concatenate(cand_scoress, axis=0).astype(np.float32, copy=False)
    n_cands_per_sample = np.asarray(n_cands_per_sample).astype(np.int32, copy=False)

    return cand_features, n_cands_per_sample, cand_choices, cand_scoress


def padding(output, n_vars_per_sample, fill=-1e8):
    n_vars_max = tf.reduce_max(n_vars_per_sample)

    output = tf.split(
        value=output,
        num_or_size_splits=n_vars_per_sample,
        axis=1,
    )
    output = tf.concat([
        tf.pad(
            x,
            paddings=[[0, 0], [0, n_vars_max - tf.shape(x)[1]]],
            mode='CONSTANT',
            constant_values=fill)
        for x in output
    ], axis=0)

    return output

def process(policy, dataloader, top_k):
    mean_kacc = np.zeros(len(top_k))

    n_samples_processed = 0
    for batch in dataloader:

        if policy['type'] == 'gcnn':
            c, ei, ev, v, n_cs, n_vs, n_cands, cands, best_cands, cand_scores = batch

            pred_scores = policy['model']((c, ei, ev, v, tf.reduce_sum(n_cs, keepdims=True), tf.reduce_sum(n_vs, keepdims=True)), tf.convert_to_tensor(False),(4),(0))

            # filter candidate variables
            pred_scores = tf.expand_dims(tf.gather(tf.squeeze(pred_scores, 0), cands), 0)

        elif policy['type'] == 'ml-competitor':
            cand_feats, n_cands, best_cands, cand_scores = batch

            # move to numpy
            cand_feats = cand_feats.numpy()
            n_cands = n_cands.numpy()

            # feature normalization
            cand_feats = (cand_feats - policy['feat_shift']) / policy['feat_scale']

            pred_scores = policy['model'].predict(cand_feats)

            # move back to TF
            pred_scores = tf.convert_to_tensor(pred_scores.reshape((1, -1)), dtype=tf.float32)

        # padding
        pred_scores = padding(pred_scores, n_cands)
        true_scores = padding(tf.reshape(cand_scores, (1, -1)), n_cands)
        true_bestscore = tf.reduce_max(true_scores, axis=-1, keepdims=True)

        assert all(true_bestscore.numpy() == np.take_along_axis(true_scores.numpy(), best_cands.numpy().reshape((-1, 1)), axis=1))

        kacc = []
        for k in top_k:
            pred_top_k = tf.nn.top_k(pred_scores, k=k)[1].numpy()
            pred_top_k_true_scores = np.take_along_axis(true_scores.numpy(), pred_top_k, axis=1)
            kacc.append(np.mean(np.any(pred_top_k_true_scores == true_bestscore.numpy(), axis=1)))
        kacc = np.asarray(kacc)

        batch_size = int(n_cands.shape[0])
        mean_kacc += kacc * batch_size
        n_samples_processed += batch_size

    mean_kacc /= n_samples_processed

    return mean_kacc

def exp_main(args):
    print(f"problem: {args.problem}")
    print(f"gpu: {args.gpu}")
    print(f"sampling: {args.sampling}")

    os.makedirs("results", exist_ok=True)
    # result_file = f"results/{args.problem}_validation_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    seeds = eval(args.seeds) # TODO
    # seeds = [0, 1]
    gcnn_models = ['baseline']
    # gcnn_models = []
    # other_models = ['lambdamart_khalil']
    # other_models = ['extratrees_gcnn_agg']
    other_models = []
    test_batch_size = 16
    top_k = [1, 3, 5, 10]

    # if args.problem == 'setcover':
    #     gcnn_models += ['mean_convolution', 'no_prenorm']

    # problem_folders = {
    #     'setcover': f'setcover/500r_1000c_0.05d({args.sampling})/{args.sample_seed}',
    #     'cauctions': f'cauctions/100_500({args.sampling})/{args.sample_seed}',
    #     'facilities': f'facilities/100_100_5({args.sampling})/{args.sample_seed}',
    #     'indset': f'indset/500_4({args.sampling})/{args.sample_seed}',
    # }
    # problem_folder = problem_folders[args.problem]
    # test_files = list(pathlib.Path(f"data/samples/{problem_folder}/test").glob('sample_*.pkl'))

    test_files = list(pathlib.Path(f"data/samples/{args.problem}/test").glob('sample_*.pkl'))
    test_files = [str(x) for x in test_files]

    result_file = f"results/{args.problem}/{args.problem}_{args.sampling}_ss{args.sample_seed}_test_{time.strftime('%Y%m%d-%H%M%S')}.csv"

    trained_model_path = f"trained_models/{args.problem}/{args.sampling}/ss{args.sample_seed}"

    # os.makedirs('results', exist_ok=True)

    ### TENSORFLOW SETUP ### 
    if args.gpu == -1:
        tf.config.set_visible_devices(tf.config.list_physical_devices('CPU')[0])
    else:
        cpu_devices = tf.config.list_physical_devices('CPU') 
        gpu_devices = tf.config.list_physical_devices('GPU') 
        tf.config.set_visible_devices([cpu_devices[0], gpu_devices[args.gpu]])
        tf.config.experimental.set_memory_growth(gpu_devices[args.gpu], True)

    

    print(f"{len(test_files)} test samples")

    evaluated_policies = [['gcnn', model] for model in gcnn_models] + \
            [['ml-competitor', model] for model in other_models]

    fieldnames = [
        'policy',
        'seed',
    ] + [
        f'acc@{k}' for k in top_k
    ]
    os.makedirs(f'results/{args.problem}', exist_ok=True)
    with open(result_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for policy_type, policy_name in evaluated_policies:
            print(f"{policy_type}:{policy_name}...")
            for train_seed in seeds:
                rng = np.random.default_rng(train_seed)
                tf.random.set_seed(rng.integers(np.iinfo(int).max))

                policy = {}
                policy['name'] = policy_name
                policy['type'] = policy_type

                if policy['type'] == 'gcnn':
                    # load model
                    # sys.path.insert(0, os.path.abspath(f"models/{policy['name']}"))
                    model = importlib.import_module(f"Gasse_l2b.models.{policy['name']}.model")
                    # policy['model'] = model.GCNPolicy()
                    
                    policy['model'] = model.GCNPolicy()
                    # TODO
                    # policy['model'].restore_state(f"trained_models/{args.problem}/{policy['name']}/{seed}/best_params.pkl")
                    policy['model'].restore_state(f"{trained_model_path}/ts{train_seed}/best_params.pkl")
                    # policy['model'].call = tfe.defun(policy['model'].call, input_signature=policy['model'].input_signature)
                    policy['batch_datatypes'] = [tf.float32, tf.int32, tf.float32,
                            tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.float32]
                    policy['batch_fun'] = load_batch_gcnn
                else:
                    # load feature normalization parameters
                    try:
                        with open(f"trained_models/{args.problem}/{policy['name']}/{train_seed}/normalization.pkl", 'rb') as f:
                            policy['feat_shift'], policy['feat_scale'] = pickle.load(f)
                    except:
                            policy['feat_shift'], policy['feat_scale'] = 0, 1

                    # load model
                    if policy_name.startswith('svmrank'):
                        policy['model'] = svmrank.Model().read(f"trained_models/{args.problem}/{policy['name']}/{train_seed}/model.txt")
                    else:
                        with open(f"trained_models/{args.problem}/{policy['name']}/{train_seed}/model.pkl", 'rb') as f:
                            # try:
                            #     policy['model'] = pickle.load(f)
                            # except EOFError:
                            #     print('none')
                                   
                            policy['model'] = pickle.load(f)

                    # load feature specifications
                    with open(f"trained_models/{args.problem}/{policy['name']}/{train_seed}/feat_specs.pkl", 'rb') as f:
                        # try:
                        #         feat_specs = pickle.load(f)
                        # except EOFError:
                        #         print('none')
                        feat_specs = pickle.load(f)

                    policy['batch_datatypes'] = [tf.float32, tf.int32, tf.int32, tf.float32]
                    policy['batch_fun'] = lambda x: load_batch_flat(x, feat_specs['type'], feat_specs['augment'], feat_specs['qbnorm'])

                test_data = tf.data.Dataset.from_tensor_slices(test_files)
                test_data = test_data.batch(test_batch_size)
                test_data = test_data.map(lambda x: tf.py_function(
                    policy['batch_fun'], [x], policy['batch_datatypes']))
                
                # test_data = test_data.map
                
                test_data = test_data.prefetch(2)

                test_kacc = process(policy, test_data, top_k)
                print(f"  {train_seed} " + " ".join([f"acc@{k}: {100*acc:4.1f}" for k, acc in zip(top_k, test_kacc)]))

                writer.writerow({
                    **{
                        'policy': f"{policy['type']}:{policy['name']}",
                        'seed': train_seed,
                    },
                    **{
                        f'acc@{k}': test_kacc[i] for i, k in enumerate(top_k)
                    },
                })
                csvfile.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--sampling',
        help='Sampling Strategy',
        choices=['uniform5', 'depthK', 'depthK2'],
        default='uniform5'
    )
    parser.add_argument(
        '-s', '--seeds',
        help='Random generator seeds as a python list or range representation.',
        # type=utilities.valid_seed,
        default="range(0,5)",
    )
    parser.add_argument(
        '--sample_seed',
        help='seed of the sampled data',
        choices=['uniform5', 'depthK', 'depthK2'],
        default='uniform5'
    )
    args = parser.parse_args()

    exp_main(args)