"""
https://towardsdatascience.com/drl-01-a-gentle-introduction-to-deep-reinforcement-learning-405b79866bf4
"""
import gym

env = gym.make('FrozenLake-v0', is_slippery=False)  # Deterministic
env.reset()  # reset to the initial state

actions = {'Left': 0, 'Down': 1, 'Right': 2, 'Up': 3}


def vis_attrs():
    print(env.action_space)  # Discrete(4) ==> 0 Left, 1 Down, 2 Right, 3 Up
    print(env.observation_space)  # Discrete(16) 观测 state 有 16 位置
    # env.render()  # 显示当前 state, 展示棋盘
    """ S:START, G:GOAL, H:HOLE, F:FROZEN
    SFFF
    FHFH
    FFFH
    HFFG
    """
    print(env.env.P)


def random_play_frozen_lake():
    # 随机选 action，实测发现，很容易掉坑里
    while True:
        reward = 0.
        for t in range(100):
            print("\nTimestep {}".format(t))

            a = env.action_space.sample()  # 随机选择 a，虽然选定了1个 a，但有 33% 概率滑向 左右
            observation, reward, done, _ = env.step(a)
            print(f'action: {a}, observation: {observation}, reward: {reward}')
            env.render()  # 显示当前 state

            if done:
                print("\nEpisode terminated early")
                break

        if reward > 0:
            break


def play_good_plan():
    good_plan = ['Right'] * 2 + ['Down'] * 3 + ['Right']  # 行为 list

    for a in good_plan:
        new_state, reward, done, info = env.step(actions[a])
        print(f'\naction: {a}, observation: {new_state}, reward: {reward}')
        env.render()
        if done:  # reward = 1.0
            print("\nEpisode terminated early")
            break


if __name__ == '__main__':
    play_good_plan()
