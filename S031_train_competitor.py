import pickle
import os
import argparse
import numpy as np
import utilities
import pathlib
import tensorflow as tf

from utilities import log,load_flat_samples


def load_samples(filenames, feat_type, label_type, augment, qbnorm, size_limit, logfile=None):
    x, y, ncands = [], [], []
    total_ncands = 0

    for i, filename in enumerate(filenames):
        cand_x, cand_y, best = load_flat_samples(filename, feat_type, label_type, augment, qbnorm)

        x.append(cand_x)
        y.append(cand_y)
        ncands.append(cand_x.shape[0])
        total_ncands += ncands[-1]

        if (i + 1) % 100 == 0:
            log(f"  {i+1}/{len(filenames)} files processed ({total_ncands} candidate variables)", logfile)

        if total_ncands >= size_limit:
            log(f"  dataset size limit reached ({size_limit} candidate variables)", logfile)
            break

    x = np.concatenate(x)
    y = np.concatenate(y)
    ncands = np.asarray(ncands)

    if total_ncands > size_limit:
        x = x[:size_limit]
        y = y[:size_limit]
        ncands[-1] -= total_ncands - size_limit

    return x, y, ncands


def exp_main(args):
    feats_type = 'nbr_maxminmean'
    sampling_strategy = args.sampling
    seeds = eval(args.seeds)
    if args.gpu == -1:
        tf.config.set_visible_devices(tf.config.list_physical_devices('CPU')[0])
    else:
        cpu_devices = tf.config.list_physical_devices('CPU') 
        gpu_devices = tf.config.list_physical_devices('GPU') 
        tf.config.set_visible_devices([cpu_devices[0], gpu_devices[args.gpu]])
        tf.config.experimental.set_memory_growth(gpu_devices[args.gpu], True)
    for seed in seeds:
        problem_folders = {
            'setcover': f'setcover/500r_1000c_0.05d({sampling_strategy})/{args.sample_seed}',
            'cauctions': f'cauctions/100_500({sampling_strategy})/{args.sample_seed}',
            'facilities': f'facilities/100_100_5({sampling_strategy})/{args.sample_seed}', # TODO
            'indset': f'indset/500_4({sampling_strategy})/{args.sample_seed}',
        }
        problem_folder = problem_folders[args.problem]


        if args.model == 'extratrees':
            train_max_size = 2500
            valid_max_size = 2500
            feat_type = 'gcnn_agg'
            feat_qbnorm = False
            feat_augment = False
            label_type = 'scores'

        elif args.model == 'lambdamart':
            train_max_size = 2500
            valid_max_size = 2500
            feat_type = 'khalil'
            feat_qbnorm = True
            feat_augment = False
            label_type = 'bipartite_ranks'


        rng = np.random.default_rng(seeds)

        running_dir = f"trained_models/{args.problem}/{args.model}_{feat_type}/{sampling_strategy}/ts{seed}"

        os.makedirs(running_dir)

        logfile = f"{running_dir}/log.txt"
        log(f"Logfile for {args.model} model on {args.problem} with seed {args.seeds}", logfile)

        # Data loading
        train_files = list(pathlib.Path(f'data/samples/{problem_folder}/train').glob('sample_*.pkl'))
        valid_files = list(pathlib.Path(f'data/samples/{problem_folder}/valid').glob('sample_*.pkl'))

        log(f"{len(train_files)} training files", logfile)
        log(f"{len(valid_files)} validation files", logfile)

        log("Loading training samples", logfile)
        train_x, train_y, train_ncands = load_samples(
                rng.permutation(train_files),
                feat_type, label_type, feat_augment, feat_qbnorm,
                train_max_size, logfile)
        log(f"  {train_x.shape[0]} training samples", logfile)

        log("Loading validation samples", logfile)
        valid_x, valid_y, valid_ncands = load_samples(
                valid_files,
                feat_type, label_type, feat_augment, feat_qbnorm,
                valid_max_size, logfile)
        log(f"  {valid_x.shape[0]} validation samples", logfile)

        # Data normalization
        log("Normalizing datasets", logfile)
        x_shift = train_x.mean(axis=0)
        x_scale = train_x.std(axis=0)
        x_scale[x_scale == 0] = 1

        valid_x = (valid_x - x_shift) / x_scale
        train_x = (train_x - x_shift) / x_scale

        # Saving feature parameters
        with open(f"{running_dir}/feat_specs.pkl", "wb") as file:
            pickle.dump({
                    'type': feat_type,
                    'augment': feat_augment,
                    'qbnorm': feat_qbnorm,
                }, file)

        # save normalization parameters
        with open(f"{running_dir}/normalization.pkl", "wb") as f:
            pickle.dump((x_shift, x_scale), f)

        log("Starting training", logfile)
        if args.model == 'extratrees':
            from sklearn.ensemble import ExtraTreesRegressor

            # Training
            model = ExtraTreesRegressor(
                n_estimators=100,
                random_state=rng.integers(100),)
            model.verbose = True
            model.fit(train_x, train_y)
            model.verbose = False

            # Saving model
            with open(f"{running_dir}/model.pkl", "wb") as file:
                pickle.dump(model, file)

            # Testing
            loss = np.mean((model.predict(valid_x) - valid_y) ** 2)
            log(f"Validation RMSE: {np.sqrt(loss):.2f}", logfile)

        elif args.model == 'lambdamart':
            import pyltr
            train_qids = np.repeat(np.arange(len(train_ncands)), train_ncands)
            valid_qids = np.repeat(np.arange(len(valid_ncands)), valid_ncands)

            # Training
            model = pyltr.models.LambdaMART(verbose=1, random_state=rng.integers(100), n_estimators=500)
            model.fit(train_x, train_y, train_qids,
            monitor=pyltr.models.monitors.ValidationMonitor(
                    valid_x, valid_y, valid_qids, metric=model.metric))

            # Saving model
            with open(f"{running_dir}/model.pkl", "wb") as file:
                pickle.dump(model, file)

            # Testing
            loss = model.metric.calc_mean(valid_qids, valid_y, model.predict(valid_x))
            log(f"Validation log-NDCG: {np.log(loss)}", logfile)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    parser.add_argument(
        '-m', '--model',
        help='Model to be trained.',
        type=str,
        choices=['svmrank', 'extratrees', 'lambdamart'],
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-s', '--seeds',
        help='Random generator seeds as a python list or range representation.',
        # type=utilities.valid_seed,
        default=0,
    )
    parser.add_argument(
        '--sampling',
        help='Sampling Strategy',
        choices=['uniform5', 'depthK', 'depthK2'],
        default='uniform5'
    )
    parser.add_argument(
        '--sample_seed',
        help='seed of the sampled data',
        type=utilities.valid_seed,
        default=0
    )
    args = parser.parse_args()
    exp_main(args)

