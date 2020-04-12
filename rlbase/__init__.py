# Algorithms
from rlbase.algos.ddpg.ddpg import ddpg as ddpg
from rlbase.algos.ppo.ppo import ppo as ppo
from rlbase.algos.sac.sac import sac as sac
from rlbase.algos.td3.td3 import td3 as td3
from rlbase.algos.vpg.vpg import vpg as vpg
from rlbase.algos.reinforce.reinforce import reinforce as reinforce
from rlbase.algos.dqn.dqn import dqn as dqn
from rlbase.algos.her.her_dqn import her_dqn as her_dqn
from rlbase.algos.her.her_td3 import her_td3 as her_td3

# Loggers
from rlbase.utils.logx import Logger, EpochLogger

# Version
from rlbase.version import __version__

# Register BitFlip environment
from gym.envs.registration import register

register(
    id='BitFlip-v0',
    entry_point='rlbase.envs:BitFlip',
)
