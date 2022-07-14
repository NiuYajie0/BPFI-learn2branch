import os
import argparse
import multiprocessing as mp
import pickle
import glob
import numpy as np
import shutil
import gzip
from numpy.core.numeric import NaN

import pyscipopt as scip
import utilities
import pandas as pd
import time

class SamplingAgent(scip.Branchrule):

    def __init__(self, episode, instance, seed, out_queue, exploration_policy, query_expert_prob, out_dir, trueOutDir, problem, lock, depthDict_accessTimes, depthDict_sampleTimes, follow_expert=True, samplingStrategy='uniform5', sampleForTraining=True):
        self.episode = episode
        self.instance = instance
        self.seed = seed
        self.out_queue = out_queue
        self.exploration_policy = exploration_policy
        self.query_expert_prob = query_expert_prob
        self.out_dir = out_dir
        self.follow_expert = follow_expert
        self.visited_maxDepth = 0
        self.problem = problem

        self.rng = np.random.default_rng(seed)
        self.new_node = True
        self.sample_counter = 0
        self.depthK = 10
        self.NNodeK = 20

        self.samplingStrategy = samplingStrategy
        self.sampleForTraining = sampleForTraining
        self.trueOutDir = trueOutDir
        self.SolStatsSummary_file = 'AllEpsSolStats.csv'

        self.lock = lock
        self.depthDict_accessTimes = depthDict_accessTimes
        self.depthDict_sampleTimes = depthDict_sampleTimes
        # self.depthTable = depthTable

    def branchinit(self):
        self.khalil_root_buffer = {}

    def branchexitsol(self):
        df = pd.DataFrame()
        df['episode'] = [self.episode]
        df['instance'] = [self.instance]
        df['visitedNNodes'] = [self.model.getNNodes()]
        df['maxDepth'] = [self.visited_maxDepth]

        with open(f"{self.trueOutDir}/{self.SolStatsSummary_file}", 'a') as f:
            df.to_csv(f, header=False)

    def branchexeclp(self, allowaddcons):

        if self.model.getNNodes() == 1:
            # initialize root buffer for Khalil features extraction
            utilities.extract_khalil_variable_features(self.model, [], self.khalil_root_buffer)

        depth = self.model.getDepth()
        if self.visited_maxDepth < depth:
            self.visited_maxDepth = depth

        # once in a while, also run the expert policy and record the (state, action) pair
        if not self.sampleForTraining: # if generating valid set or test set
            query_expert = (self.rng.random() < self.query_expert_prob)
        else:
            with self.lock:
                if depth not in self.depthDict_accessTimes:
                    self.depthDict_accessTimes[depth] = 1
                else:
                    self.depthDict_accessTimes[depth] += 1

            if self.samplingStrategy == 'uniform5': 
                query_expert = (self.rng.random() < self.query_expert_prob)
            elif self.samplingStrategy == 'depthK':
                query_expert = (self.rng.random() < self.query_expert_prob) or ((depth < self.depthK) and (self.model.getNNodes() <= self.NNodeK))
            elif self.samplingStrategy == 'depthK2':

                with self.lock:
                    if depth not in self.depthDict_sampleTimes:
                        query_expert = True
                    else:
                        valSum = sum(self.depthDict_sampleTimes.values())
                        scores = {k:valSum/v for k,v in self.depthDict_sampleTimes.items()}
                        query_expert_prob = scores[depth] / sum(scores.values())
                        query_expert = (self.rng.random() < query_expert_prob)
            elif self.samplingStrategy == 'depthK3':

                with self.lock:
                    if depth not in self.depthDict_sampleTimes:
                        query_expert_prob = 0.5
                    else:
                        valSum = sum(self.depthDict_sampleTimes.values())
                        scores = {k:valSum/v for k,v in self.depthDict_sampleTimes.items()}
                        query_expert_prob = scores[depth] / sum(scores.values())
                        query_expert_prob = min(max(0.05, query_expert_prob), 0.5)
                    query_expert = (self.rng.random() < query_expert_prob)
            else:
                raise ValueError("Argument samplingStrategy can only be chosen from ['uniform5', 'depthK', 'depthK2', 'depthK3']")

        if query_expert:
            # global lck, depthTable
            if self.sampleForTraining:
                with self.lock:
                    if depth not in self.depthDict_sampleTimes:
                        self.depthDict_sampleTimes[depth] = 1
                    else:
                        self.depthDict_sampleTimes[depth] += 1

            state = utilities.extract_state(self.model)
            cands, *_ = self.model.getPseudoBranchCands()
            state_khalil = utilities.extract_khalil_variable_features(self.model, cands, self.khalil_root_buffer)

            result = self.model.executeBranchRule('vanillafullstrong', allowaddcons)
            cands_, scores, npriocands, bestcand = self.model.getVanillafullstrongData()


            assert result == scip.SCIP_RESULT.DIDNOTRUN
            assert all([c1.getCol().getLPPos() == c2.getCol().getLPPos() for c1, c2 in zip(cands, cands_)])

            action_set = [c.getCol().getLPPos() for c in cands]
            expert_action = action_set[bestcand]

            data = [state, state_khalil, expert_action, action_set, scores]

            # Do not record inconsistent scores. May happen if SCIP was early stopped (time limit).
            if not any([s < 0 for s in scores]):

                filename = f'{self.out_dir}/sample_{self.episode}_{self.sample_counter}.pkl'
                with gzip.open(filename, 'wb') as f:
                    pickle.dump({
                        'episode': self.episode,
                        'instance': self.instance,
                        'seed': self.seed,
                        'node_number': self.model.getCurrentNode().getNumber(),
                        'node_depth': self.model.getCurrentNode().getDepth(),
                        'data': data,
                        }, f)

                self.out_queue.put({
                    'type': 'sample',
                    'episode': self.episode,
                    'instance': self.instance,
                    'seed': self.seed,
                    'node_number': self.model.getCurrentNode().getNumber(),
                    'node_depth': self.model.getCurrentNode().getDepth(),
                    'filename': filename,
                })

                self.sample_counter += 1

        # if exploration and expert policies are the same, prevent running it twice
        if not query_expert or (not self.follow_expert and self.exploration_policy != 'vanillafullstrong'):
            result = self.model.executeBranchRule(self.exploration_policy, allowaddcons)

        # apply 'vanillafullstrong' branching decision if needed
        if query_expert and self.follow_expert or self.exploration_policy == 'vanillafullstrong':
            assert result == scip.SCIP_RESULT.DIDNOTRUN
            cands, scores, npriocands, bestcand = self.model.getVanillafullstrongData()
            self.model.branchVar(cands[bestcand])
            result = scip.SCIP_RESULT.BRANCHED

        return {"result": result}


def make_samples(in_queue, out_queue, samplingStrategy, trueOutDir, problem, forTraining, lock, depthDict_accessTimes, depthDict_sampleTimes):
    """
    Worker loop: fetch an instance, run an episode and record samples.

    Parameters
    ----------
    in_queue : multiprocessing.Queue
        Input queue from which orders are received.
    out_queue : multiprocessing.Queue
        Output queue in which to send samples.
    """

    while True:
        episode, instance, seed, exploration_policy, query_expert_prob, time_limit, out_dir = in_queue.get()
        print(f'[w {os.getpid()}] episode {episode}, seed {seed}, processing instance \'{instance}\'...')

        m = scip.Model()
        m.setIntParam('display/verblevel', 0)
        m.readProblem(f'{instance}')
        utilities.init_scip_params(m, seed=seed)
        m.setIntParam('timing/clocktype', 2)
        m.setRealParam('limits/time', time_limit)

        branchrule = SamplingAgent(
            episode=episode,
            instance=instance,
            seed=seed,
            out_queue=out_queue,
            exploration_policy=exploration_policy,
            query_expert_prob=query_expert_prob,
            out_dir=out_dir,
            trueOutDir=trueOutDir,
            problem=problem,
            samplingStrategy=samplingStrategy,
            sampleForTraining=forTraining,
            lock = lock,
            depthDict_accessTimes = depthDict_accessTimes,
            depthDict_sampleTimes = depthDict_sampleTimes,
            )

        m.includeBranchrule(
            branchrule=branchrule,
            name="Sampling branching rule", desc="",
            priority=666666, maxdepth=-1, maxbounddist=1)

        m.setBoolParam('branching/vanillafullstrong/integralcands', True)
        m.setBoolParam('branching/vanillafullstrong/scoreall', True)
        m.setBoolParam('branching/vanillafullstrong/collectscores', True)
        m.setBoolParam('branching/vanillafullstrong/donotbranch', True)
        m.setBoolParam('branching/vanillafullstrong/idempotent', True)

        out_queue.put({
            'type': 'start',
            'episode': episode,
            'instance': instance,
            'seed': seed,
        })

        m.optimize()
        m.freeProb()

        print(f"[w {os.getpid()}] episode {episode} done, {branchrule.sample_counter} samples")

        out_queue.put({
            'type': 'done',
            'episode': episode,
            'instance': instance,
            'seed': seed,
        })


def send_orders(orders_queue, instances, seed, exploration_policy, query_expert_prob, time_limit, out_dir):
    """
    Continuously send sampling orders to workers (relies on limited
    queue capacity).

    Parameters
    ----------
    orders_queue : multiprocessing.Queue
        Queue to which to send orders.
    instances : list
        Instance file names from which to sample episodes.
    seed : int
        Random seed for reproducibility.
    exploration_policy : str
        Branching strategy for exploration.
    query_expert_prob : float in [0, 1]
        Probability of running the expert strategy and collecting samples.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    out_dir: str
        Output directory in which to write samples.
    """
    rng = np.random.default_rng(seed)

    episode = 0
    while True:
        instance = rng.choice(instances)
        seed = rng.integers(2**32)
        orders_queue.put([episode, instance, seed, exploration_policy, query_expert_prob, time_limit, out_dir])
        episode += 1


def collect_samples(instances, out_dir, rng, n_samples, n_jobs,
                    exploration_policy, query_expert_prob, time_limit, samplingStrategy, problem, lock, depthDict_accessTimes, depthDict_sampleTimes, forTraining=True):
    """
    Runs branch-and-bound episodes on the given set of instances, and collects
    randomly (state, action) pairs from the 'vanilla-fullstrong' expert
    brancher.

    Parameters
    ----------
    instances : list
        Instance files from which to collect samples.
    out_dir : str
        Directory in which to write samples.
    rng : numpy.random.RandomState
        A random number generator for reproducibility.
    n_samples : int
        Number of samples to collect.
    n_jobs : int
        Number of jobs for parallel sampling.
    exploration_policy : str
        Exploration policy (branching rule) for sampling.
    query_expert_prob : float in [0, 1]
        Probability of using the expert policy and recording a (state, action)
        pair.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    """

    startTime = time.time()

    os.makedirs(out_dir, exist_ok=True)

    samplingStatsSummary_file = 'AllEpsSolStats.csv'
    df = pd.DataFrame()
    df['episode'] = []
    df['instance'] = []
    df['visitedNNodes'] = []
    df['maxDepth'] = []
    df.to_csv(f'{out_dir}/{samplingStatsSummary_file}')

    # start workers
    orders_queue = mp.Queue(maxsize=2*n_jobs)
    answers_queue = mp.SimpleQueue()
    workers = []
    for i in range(n_jobs):
        p = mp.Process(
                target=make_samples,
                args=(orders_queue, answers_queue, samplingStrategy, out_dir, problem, forTraining, lock, depthDict_accessTimes, depthDict_sampleTimes),
                daemon=True)
        workers.append(p)
        p.start()

    tmp_samples_dir = f'{out_dir}/tmp'
    os.makedirs(tmp_samples_dir, exist_ok=True)

    # start dispatcher
    dispatcher = mp.Process(
            target=send_orders,
            args=(orders_queue, instances, rng.integers(2**32), exploration_policy, query_expert_prob, time_limit, tmp_samples_dir),
            daemon=True)
    dispatcher.start()

    # record answers and write samples
    buffer = {}
    current_episode = 0
    i = 0
    in_buffer = 0
    while i < n_samples:
        status = answers_queue.get()

        # add received sample to buffer
        if status['type'] == 'start':
            buffer[status['episode']] = []
        else:
            buffer[status['episode']].append(status)
            if status['type'] == 'sample':
                in_buffer += 1

        # if any, write samples from current episode
        while current_episode in buffer and buffer[current_episode]:
            samples_to_write = buffer[current_episode]
            buffer[current_episode] = []

            for sample in samples_to_write:

                # if no more samples here, move to next episode
                if sample['type'] == 'done':
                    del buffer[current_episode]
                    current_episode += 1

                # else write sample
                else:


                    # in SamplingAgent.branchexeclp: filename = f'{self.out_dir}/sample_{self.episode}_{self.sample_counter}.pkl'
                    # now changed to: f'{out_dir}/sample_{i+1}.pkl'
                    os.rename(sample['filename'], f'{out_dir}/sample_{i+1}.pkl')
                    in_buffer -= 1
                    i += 1
                    print(f"[m {os.getpid()}] {i} / {n_samples} samples written, ep {sample['episode']} ({in_buffer} in buffer).")

                    # early stop dispatcher (hard)
                    if in_buffer + i >= n_samples and dispatcher.is_alive():
                        dispatcher.terminate()
                        print(f"[m {os.getpid()}] dispatcher stopped...")

                    # as soon as enough samples are collected, stop
                    if i == n_samples:
                        buffer = {}
                        break

    # stop all workers (hard)
    for p in workers:
        p.terminate()

    shutil.rmtree(tmp_samples_dir, ignore_errors=True)

    endTime = time.time()
    with open(f'{out_dir}/elapseTime={endTime - startTime}.txt', 'w') as fp:
        pass


def exp_main(args):
    samplingStrategy = args.sampling
    print(f"seed {args.seed}")
    train_size, valid_size, test_size = eval(args.n_samples)
    exploration_strategy = 'pscost'
    node_record_prob = 0.05
    time_limit = 3600

    if args.problem == 'setcover':
        instances_train = glob.glob('data/instances/setcover/train_500r_1000c_0.05d/*.lp')
        instances_valid = glob.glob('data/instances/setcover/valid_500r_1000c_0.05d/*.lp')
        instances_test = glob.glob('data/instances/setcover/test_500r_1000c_0.05d/*.lp')
        out_dir = f'data/samples/setcover/500r_1000c_0.05d({samplingStrategy})/{args.seed}'

    elif args.problem == 'cauctions':
        instances_train = glob.glob('data/instances/cauctions/train_100_500/*.lp')
        instances_valid = glob.glob('data/instances/cauctions/valid_100_500/*.lp')
        instances_test = glob.glob('data/instances/cauctions/test_100_500/*.lp')
        out_dir = f'data/samples/cauctions/100_500({samplingStrategy})/{args.seed}'

    elif args.problem == 'indset':
        instances_train = glob.glob('data/instances/indset/train_500_4/*.lp')
        instances_valid = glob.glob('data/instances/indset/valid_500_4/*.lp')
        instances_test = glob.glob('data/instances/indset/test_500_4/*.lp')
        out_dir = f'data/samples/indset/500_4({samplingStrategy})/{args.seed}'

    elif args.problem == 'facilities':
        instances_train = glob.glob('data/instances/facilities/train_100_100_5/*.lp')
        instances_valid = glob.glob('data/instances/facilities/valid_100_100_5/*.lp')
        instances_test = glob.glob('data/instances/facilities/test_100_100_5/*.lp')
        out_dir = f'data/samples/facilities/100_100_5({samplingStrategy})/{args.seed}'
        time_limit = 600

    else:
        raise NotImplementedError

    print(f"{len(instances_train)} train instances for {train_size} samples")
    print(f"{len(instances_valid)} validation instances for {valid_size} samples")
    print(f"{len(instances_test)} test instances for {test_size} samples")

    # create output directory, throws an error if it already exists
    os.makedirs(out_dir)

    
    lck = mp.Lock()
    
    depthDict_accessTimes = mp.Manager().dict()
    depthDict_sampleTimes = mp.Manager().dict()
    # ns._depthTable = pd.DataFrame(columns=['accessTimes', 'sampleTimes'])

    rng = np.random.default_rng(args.seed)
    collect_samples(instances_train, out_dir + '/train', rng, train_size,
                    args.njobs, exploration_policy=exploration_strategy,
                    query_expert_prob=node_record_prob,
                    time_limit=time_limit, samplingStrategy=samplingStrategy, problem=args.problem, forTraining=True, lock=lck, depthDict_accessTimes=depthDict_accessTimes, depthDict_sampleTimes=depthDict_sampleTimes)
    depthTable = pd.concat([pd.Series(depthDict_accessTimes), pd.Series(depthDict_sampleTimes)], axis=1)
    depthTable.columns = ['accessTimes', 'sampleTimes']
    depthTable.to_csv(f'{out_dir}/depthTable(trainSol).csv')

    rng = np.random.default_rng(args.seed + 1)
    collect_samples(instances_valid, out_dir + '/valid', rng, test_size,
                    args.njobs, exploration_policy=exploration_strategy,
                    query_expert_prob=node_record_prob,
                    time_limit=time_limit, samplingStrategy=samplingStrategy, problem=args.problem, forTraining=False,lock=lck, depthDict_accessTimes=depthDict_accessTimes,depthDict_sampleTimes=depthDict_sampleTimes)

    # TODO 1. 不管什么采样方式，测试集都应该是同一个测试集，所以应该检查有没有一个（公用的）测试集，如果有的话就不生成测试集了；
    # TODO 2. 到时在 S04_test.py 那里再修改一下，不管什么sampling strategy，都应该用那个公用的测试数据集

    sharedTestSet_path = f'data/samples/{args.problem}/test'
    if not os.path.exists(sharedTestSet_path):
        rng = np.random.default_rng(args.seed + 2)
        collect_samples(instances_test, sharedTestSet_path, rng, test_size,
                        args.njobs, exploration_policy=exploration_strategy,
                        query_expert_prob=node_record_prob,
                        time_limit=time_limit, samplingStrategy=samplingStrategy, problem=args.problem, forTraining=False, lock=lck, depthDict_accessTimes=depthDict_accessTimes,depthDict_sampleTimes=depthDict_sampleTimes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=utilities.valid_seed,
        default=0,
    )
    parser.add_argument(
        '-j', '--njobs',
        help='Number of parallel jobs.',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--sampling',
        help='Sampling Strategy',
        choices=['uniform5', 'depthK', 'depthK2'],
        default='uniform5'
    )
    parser.add_argument(
        '-n', '--n_samples',
        help='Number of generated n_samples as (train_size, valid_size, test_size).',
        default="(1000, 200, 200)",
    )
    args = parser.parse_args()
    start = time.time()
    exp_main(args)
    end = time.time()
    print(end - start)