#  re-sln

Re-implementation of the paper titled "Noise against noise: stochastic label noise helps combat inherent label noise" from ICLR 2021.

## Setup

Make a virtual env and isntall dependencies from the ```environment.yml``` file.

## Run

Run the ```main.py``` notebook.

## Logs

Tensorboard is used for logging. Share your logs as shown below (from [here](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html#share-tensorboard-dashboards)):
```
tensorboard dev upload --logdir runs --name "re-sln results" --description "By Mark"
```

## Experiments

Available models: CE, SLN, SLN+MO, SLN+MO+LC
Available noisy data sets for CIFAR-10 (p=0.4): sym (paper, mine), asym (paper, mine), dependent (paper), openset (paper, mine)

| `model` / `noise` | sym |  | asym |  | dependent |  | openset |  |
| - | - | - | - | - | - | - | - | - |
|  | paper | custom | paper | custom | paper | custom | paper | custom |
| CE | exp_2021-11-25 13:17:26.851200 | exp_2021-11-25 20:30:28.794160 | exp_2021-11-26 09:21:19.524188 | exp_2021-11-26 14:03:18.975684 | exp_2021-11-26 20:14:40.983299 | x | x | exp_2021-11-27 13:35:23.026659 |
| SLN | exp_2021-11-25 15:38:09.361059 | exp_2021-11-25 20:31:37.546765 | exp_2021-11-26 11:41:56.758060 | exp_2021-11-26 16:11:00.844488 | exp_2021-11-27 11:07:55.847340 | x | x | exp_2021-11-27 13:44:37.885816 |
|  SLN+MO | exp_2021-11-25 16:46:29.066838 | exp_2021-11-26 09:18:14.291265 | exp_2021-11-26 11:44:27.727904 | exp_2021-11-26 16:14:06.628600 | exp_2021-11-27 11:11:07.020347 | x | x | exp_2021-11-27 13:46:43.777573 |
| SLN+MO+LC | exp_2021-11-26 11:18:36.051172 | exp_2021-11-26 13:51:03.590616 | exp_2021-11-26 13:57:45.567433 | exp_2021-11-26 16:16:06.031597 | exp_2021-11-27 11:14:24.120092 | x | x | exp_2021-11-28 16:34:38.935269 |

Training Times per Computational Resources
- exp_2021-11-25 13:17:26.851200: 1 h 31 m with 1 x V100
get times and final test accs from runs/

---
Cifar100

| `model` / `noise` | sym |  | asym |  | dependent |  | openset |  |
| - | - | - | - | - | - | - | - | - |
|  | paper | custom | paper | custom | paper | custom | paper | custom |
| CE | exp_2021-11-29 13:02:42.947124 | exp_2021-11-29 15:14:24.277293  | exp_2021-12-02 17:15:05.141925 | exp_2021-12-02 20:50:30.272408 | exp_2021-12-03 12:09:50.374569 | x | x | x |
| SLN | exp_2021-11-29 13:12:28.474547 | exp_2021-11-29 15:15:36.143703 | exp_2021-12-02 17:34:08.440889 | exp_2021-12-02 20:55:53.387841 | exp_2021-12-03 14:37:51.783033 | x | x | x |
|  SLN+MO | exp_2021-11-29 13:16:11.590910 | exp_2021-11-29 22:15:08.652843 | exp_2021-12-02 17:39:34.952358 | exp_2021-12-03 11:53:37.290785 | exp_2021-12-03 14:43:27.237441 | x | x | x |
| SLN+MO+LC | exp_2021-11-29 22:04:19.910053 | exp_2021-11-29 22:26:18.532929 | exp_2021-12-02 20:43:32.204172 | exp_2021-12-03 12:01:04.662910 | exp_2021-12-03 14:51:11.441549 | x | x | x |


HP search

cifar10, sym, noise from paper: hp_2021-12-03_13-18-02 (sigma=[0.1, 0.2, 0.5, 1.0]) -> best 1.0 (good)
cifar10, sym, custom noise: hp_2021-12-04_17-04-54 (sigma=[0.1, 0.2, 0.5, 1.0]) -> best 1.0 (good)
