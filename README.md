# Batch-wise Permutation Feature Importance Evaluation and Problem-Specific Bigraph for Learn-to-Branch

Yajie Niu, Chen Peng, Bolin Liao

## Installation

This work is based on [learn2branch](https://github.com/ds4dm/learn2branch), which proposes Graph Convolutional Neural Network for learn-to-branch.

Please follow installation instructions of [learn2branch](https://github.com/ds4dm/learn2branch/blob/master/INSTALL.md) to install [SCIP](https://www.scipopt.org/) and PySCIPOpt.

## Structure and running

In the instructions below we assumed that a bash variable `PROBLEM` exists. For example,

```bash
PROBLEM=setcover
```

### Generate Instances

```bash
python Gasse_l2b/S01_generate_instances.py

```

### Generate dataset

```bash
python Gasse_l2b/S02_generate_dataset.py
```

### Train models

```bash
python S03_train_gcnn.py
```

### BPFI evaluation

```bash
python S041_evaluate_BPFI.py
```

### Train competitors

```bash
python S031_train_competitors.py
```

### Test models

```bash
python S04_test_gcnn.py
```

### Evaluate models

```bash
python S05_evaluate_gcnn.py
```

### You can use the file exp.py to run all of the above files.
