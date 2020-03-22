# RL Base

This is a fork of [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/index.html), 
an educational resource produced by OpenAI that makes it easier to learn about deep 
reinforcement learning.

This fork has all the PyTorch implementations from the original repo, plus additional algorithms.
The new implementations are all following the [Spinning Up code format](https://spinningup.openai.com/en/latest/user/algorithms.html#code-format).

## Algorithms
The following algorithms have been implemented. 

| Algorithm | Implementation | `box`              | `discrete`            | Multi Processing      |
|:----------|:--------------:|:------------------:|:---------------------:|:---------------------:|
| REINFORCE [1] |     RLBase     | :heavy_check_mark: |   :heavy_check_mark:  |   :heavy_check_mark:  |
| VPG [2, 3]    |   Spinning Up  | :heavy_check_mark: |   :heavy_check_mark:  |   :heavy_check_mark:  |
| PPO [4]       |   Spinning Up  | :heavy_check_mark: |   :heavy_check_mark:  |   :heavy_check_mark:  |
| DQN [5, 6]    |   RLBase       | :black_square_button: | :heavy_check_mark: | :black_square_button: |
| DDPG [7]      |   Spinning Up  | :heavy_check_mark: | :black_square_button: | :black_square_button: |
| TD3 [8]       |   Spinning Up  | :heavy_check_mark: | :black_square_button: | :black_square_button: |
| SAC [9]       |   Spinning Up  | :heavy_check_mark: | :black_square_button: | :black_square_button: |

**Note:** This is a work in progress and more algorithms will be added over time.

## Installation and Instructions
* You can install the package using `pip`. This installs the package and all its dependencies.
```
# From ~/rlbase
pip install -e .
```
* Follow the Spinning Up instructions for [running the experiments](https://spinningup.openai.com/en/latest/user/running.html)
and [plotting the results](https://spinningup.openai.com/en/latest/user/plotting.html). 

## References
1. [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/content/pdf/10.1007/BF00992696.pdf)
1. [Policy Gradient Methods for Reinforcement Learning with Function Approximation ](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)
1. [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
1. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
1. [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
1. [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
1. [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
1. [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)
1. [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)

## ToDos
- [ ] Add CNN option for policies 
- [ ] Add Tensorboard support
