# RL Base

This is a fork of [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/index.html), 
an educational resource produced by OpenAI that makes it easier to learn about deep 
reinforcement learning.

This fork has all the PyTorch implementations from the original repo, plus additional algorithms.
The new implementations are all following the [Spinning Up code format](https://spinningup.openai.com/en/latest/user/algorithms.html#code-format).

## Algorithms
The following algorithms have been implemented. 

| Algorithm | Implementation | `box`              | `discrete`         | Multi Processing   |
|-----------|----------------|--------------------|--------------------|--------------------|
| REINFORCE |     RLBase     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| VPG       |   Spinning Up  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| PPO       |   Spinning Up  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| DDPG      |   Spinning Up  | :heavy_check_mark: |   :white_square:   |   :white_square:   |
| TD3       |   Spinning Up  | :heavy_check_mark: |   :white_square:   |   :white_square:   |
| SAC       |   Spinning Up  | :heavy_check_mark: |   :white_square:   |   :white_square:   |

**Note:** This is a work in progress and more algorithms will be added over time.

## Installation and Instructions
* You can install the package using `pip`. This installs the package and all its dependencies.
```
# From ~/rlbase
pip install -e .
```
* Follow the [Spinning Up instructions](https://spinningup.openai.com/en/latest/user/running.html) 
for running the experiments. 

## ToDos
- [ ] Add CNN option for policies 
- [ ] Add Tensorboard support
 