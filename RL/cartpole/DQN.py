"""
Q* = State × Action -> R; GT reward 矩阵，(S, A)

Q-Learning, state-action pair
π*(s) = argmax_a Q*(s, a)  # 根据当前的 s 选择1个最优的 a

Q* 描述 reward query table is unknown, 但是 NNs are universal function approximators, we can simply create one and train it to resemble Q*.

bellman equation: https://towardsdatascience.com/the-bellman-equation-59258a0d3fa7?gi=d6c78ab8d90a

s -> s'
Q_π(s, a) = r + γ Q_π(s', π(s'))
    π(s'): s' 状态 action 选择策略 a'，得到 (s', a') pair 的 action-value

temporal difference error

δ = Q(s, a) - (r + γ max_a Q(s', a))
"""
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    input: s, the difference between the current and previous screen patches. 当前和之前的 screen 变化
    output: Q(s, left), Q(s, right) 估计 left/right 对应的收益
    """

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2, padding=0):
            return (size + 2 * padding - kernel_size) // stride + 1

        # 计算三层 conv 后的 feature map size
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        print(convh, convw)

        linear_input_size = convw * convh * 32

        self.head = nn.Linear(linear_input_size, outputs)  # output 范围没有约束

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # B,N; 2D 拉成 vector
        x = self.head(x)  # 交给 Linear 层; 类似二分类
        return x
