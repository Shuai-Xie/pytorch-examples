"""
It stores the transitions that the agent observes, allowing us to reuse this data later. 存储已有的转移，后期使用
By sampling from it randomly, the transitions that build up a batch are decorrelated. 随机采样 得到 不相关的 batch
It has been shown that this greatly stabilizes and improves the DQN training procedure. 稳定训练

For this, we’re going to need two classses:

Transition - a named tuple representing a single transition in our environment.
    It essentially maps (state, action) pairs to their (next_state, reward) result, with the state being the screen difference image as described later on.
ReplayMemory - a cyclic buffer of bounded size that holds the transitions observed recently.
    It also implements a .sample() method for selecting a random batch of transitions for training.
"""

from collections import namedtuple
import random

# 命名 tutle，添加 tuple 使用定义的 name 作为属性
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    """
    a cyclic buffer of bounded size that holds the transitions observed recently.
    循环队列，存储最近一段时刻内的 状态观测
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """save a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)  # 为了方便用 self.position 索引 list 存储位置

        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity  # cyclic buffer

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)  # 随机从 memory 中抽取 batch_size 之前的状态

    def __len__(self):
        return len(self.memory)
