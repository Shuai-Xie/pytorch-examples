"""
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

CartPole task: https://gym.openai.com/envs/CartPole-v0/
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. 平滑铰链，小车在平滑轨道行动
The system is controlled by applying a force of +1 or -1 to the cart. 通过对推车施加+1或-1的力来控制系统
A reward of +1 is provided for every timestep that the pole remains upright. 杆保持直立的每一步奖励+1
The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.
偏离垂直方向超过15度，或者推车从中心移动超过2.4个单位时，回合结束

Action set: left or right, 使得 pole 保持正立
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import gym
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.nn.functional as F

from PIL import Image
import torch
import numpy as np
import time
from utils.misc import get_curtime

# 任务描述: https://gym.openai.com/envs/CartPole-v0/
envname = 'CartPole-v0'
env = gym.make(envname).unwrapped

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def vis_env():
    env.reset()
    print(env.action_space)
    print(env.action_space.n)  # 2, {left, right}
    # print(env.render(mode='human'))  # True, 当前状态正常? game ok?
    env.render()
    time.sleep(1)  # 查看 render 图片，避免程序直接跳出，无法观测


def get_cart_location(screen_width):
    """
    获取当前 screen_width 计算 image - state 缩放因子
    并将 cart state 转化到 image 坐标系

    归一化 x_threshold = 2.4 units, cartpole 如果 左/右 移动超过 2.4 个单位，失败

    env.reset() 会随机初始化 cartpole 状态，如:
    env.state[0] = -0.04519733550638776
    """
    world_width = env.x_threshold * 2  # 2.4*2
    scale = screen_width / world_width  # image width / state width
    # 将 state 值转化到 image 坐标系
    loc = int(env.state[0] * scale + screen_width / 2.0)
    return loc


resize = T.Compose([
    T.ToPILImage(),
    T.Resize(40, interpolation=Image.CUBIC),
    T.ToTensor()
])


def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array')  # numpy, H,W,C
    screen = screen.transpose((2, 0, 1))

    # 截取 cartpole 可视有效区域
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]  # 垂直方向 0.4-0.8

    # map state to image width
    cart_location = get_cart_location(screen_width)

    view_width = int(screen_width * 0.6)
    # 可视宽度 0.6，根据 cart_location 分三种情况，切分图片
    if cart_location < view_width // 2:  # 左侧 0.3
        slice_range = slice(view_width)  # 元素切片, 左侧 [0, view_width)
    elif cart_location > (screen_width - view_width // 2):  # 右侧 0.3
        slice_range = slice(-view_width, None)  # 右侧 [view_width, screen_width)
    else:  # 中间 0.4 位置, 以 cart_location 为中心，左右切 0.3
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)

    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]  # (3, 160, 360); 400*0.4, 600*0.6

    # Convert to float, rescale, convert to torch tensor
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)


def vis_crop_screen():
    env.reset()  # 必须有，不然返回 None
    plt.figure()
    plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
               interpolation='none')
    plt.title('Example extracted screen')
    plt.show()


def random_play():
    env.reset()
    for t in range(1000):
        print("\nTimestep {}".format(t))
        env.render()
        observation, reward, done, info = env.step(env.action_space.sample())
        print(f'observation: {observation}, reward: {reward}, info: {info}, done: {done}')
        if done:  # pole 倒了 or cart 移出边界
            env.reset()
    env.close()


"""train/test"""

from RL.cartpole.DQN import DQN
from RL.cartpole.replay_memory import ReplayMemory, Transition
from itertools import count
from torch.utils.tensorboard import SummaryWriter

import torch.optim as optim
import random
import math

# 获取模型定义参数
env.reset()
_, _, screen_h, screen_w = get_screen().shape
n_actions = env.action_space.n  # 类似分类任务的 num_classes


def build_model():
    # 学习态 policy
    policy_net = DQN(screen_h, screen_w, outputs=n_actions).to(device)
    # 固定态 target
    target_net = DQN(screen_h, screen_w, outputs=n_actions).to(device)  # 使用学习的 Q 作为 Q*
    target_net.load_state_dict(policy_net.state_dict())  # Q* = Q
    target_net.eval()  # 评估模式

    return policy_net, target_net


# ε-Greedy 策略
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

steps_done = 0  # global step，决定 ε


def select_action(state):
    """
    ε-Greedy 策略 选择要采取的 action
    :param state: cur_screen - last_screen，作为 policy_net 输入
    """
    global steps_done

    # update ε, 实验次数越多，当前积累的经验中 取到最优的可能性越大
    eps_thre = EPS_END + (EPS_START - EPS_END) * (math.exp(-1. * steps_done / EPS_DECAY))
    # 从 EPS_START=0.9 指数衰减到 EPS_END=0.05, steps_done -> ∞
    # 当 steps=EPS_DECAY, eps_thre = 0.36269752499572594

    steps_done += 1

    if random.random() > eps_thre:  # 使用当前训练的 policy_net 选择 action
        with torch.no_grad():
            # t.max(1) will return largest column value (max_val, max_idx) of each row.
            # [1] 取出 idx -> 每行 state 下预测要采取的 action
            return policy_net(state).max(dim=1)[1].view(1, 1)
    else:  # 探索，随机取 action
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)  # shape [1, 1]


GAMMA = 0.999  # discount factor, bellman equation
BATCH_SIZE = 128
TARGET_UPDATE = 10  # 每 10 个 episode 使用 policy_net 更新一次 target_net; 调成 5 效果差?


def optimize_model():
    """
    1. 采样历史数据 ('state', 'action', 'next_state', 'reward')
    2. Q(s_t, a): policy_net(state) 估计各个 action 对应 value，并根据采样取出 GT action 对应的 value [predict]
    3. max_Q(s_t+1): target_net 计算 next_state 估值，结合 bellman equation 作为 GT action_value [target]
    4. huber loss (smooth l1) 回归 [loss]
    5. optimize, 注意 grad clip
    """
    if len(memory) < BATCH_SIZE:
        return

    # 1. 采样历史数据 -> batch
    transitions = memory.sample(BATCH_SIZE)

    # cvt array of Transition to Transition class, 每个 transition 的各个属性组成 4 个 list
    # ('state', 'action', 'next_state', 'reward')
    batch = Transition(*zip(*transitions))

    # state/action/reward
    state_batch = torch.cat(batch.state)  # 取样当前 state
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # next_state 如果 next_state 非终止态，取样参与训练
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    # 2.Q(s_t, a): policy_net 根据 s_t 计算 action_value, 并取出 GT action_batch 对应的 action value
    state_action_values = policy_net(state_batch).gather(1, action_batch)  # dim=1, action 维

    # 3.Q(s_t+1): target_net 计算 s_t+1 估值，并用 bellman 方程计算 s_t 的 target 估值
    next_state_values = torch.zeros(BATCH_SIZE, device=device)  # 初始化为 0，因为对于 final_next_state, reward=0.
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()  # 计算 next_state 估值

    # bellman equation
    expected_state_action_values = reward_batch + (next_state_values * GAMMA)  # 作为 target

    # 4.Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # 5.optimize model
    optimzier.zero_grad()
    loss.backward()
    # grap clip before step
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)  # grad clip
    optimzier.step()


def plt_durations():
    plt.figure(2)
    plt.clf()

    durations_t = torch.tensor(episode_durations, dtype=torch.float32)

    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    if len(durations_t) >= 100:
        # 每次截取 100 episodes 为单位，计算 mean
        means = durations_t.unfold(0, size=100, step=1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))  # 从第 100 开始记录 mean; 注意是 step=1，而不是 100，滑动窗口最后 100 项均值
        plt.plot(means.numpy())  # 2 图

    plt.pause(1e-3)
    plt.show()


if __name__ == '__main__':
    # model
    policy_net, target_net = build_model()
    print('build model..')

    # optimzier
    optimzier = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(capacity=10000)  # 存储 10000 步历史 action

    # writer
    writer = SummaryWriter(log_dir=f'runs/{envname}_{get_curtime()}')

    num_episodes = 10000
    episode_durations = []  # each episode 持续时长，越长越好

    for i_epi in range(num_episodes):
        # 每个 episode 重置环境，episode 就指的是每一轮游戏啊
        env.reset()

        # state: difference of two consecutive frames
        last_screen = get_screen()
        cur_screen = get_screen()
        state = cur_screen - last_screen  # 1,3,H,W

        # play one episode
        for t in count():
            # select and perform action
            action = select_action(state)  # one difference, one state, one action
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            # observe new state
            last_screen = cur_screen
            cur_screen = get_screen()
            next_state = cur_screen - last_screen if not done else None

            # store transitions in memory
            memory.push(state, action, next_state, reward)  # 得到一次状态转移样本

            # move to next state
            state = next_state

            # Perform one step of the optimization
            optimize_model()  # 如果存储数量不到 batch_size 不会更新

            if done:
                duration = t + 1
                print(f'episode {i_epi}, duration: {duration}')
                episode_durations.append(duration)  # 片段 持续时长
                # plt_durations()

                writer.add_scalars('Duration', {
                    'dura': duration,
                }, global_step=i_epi)

                if i_epi >= 10:
                    writer.add_scalars('Duration', {
                        'dura_mean10': np.mean(episode_durations[-10:]),
                    }, global_step=i_epi)

                break

        # update targetnet
        if i_epi % TARGET_UPDATE == 0:
            print('update target..')
            target_net.load_state_dict(policy_net.state_dict())  # 更新参数

    env.close()
