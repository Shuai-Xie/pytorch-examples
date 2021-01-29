import torch
import torch.nn as nn
"""
apply 函数

Function.apply: fowward 前向
Module.apply: apply(fn) 对所有子模块都使用 fn
"""


def turn_on_dropout(module):
    if type(module) == torch.nn.Dropout:
        module.train()


def turn_off_dropout(module):
    if type(module) == torch.nn.Dropout:
        module.eval()


def apply_dropout(model):
    model.eval()
    model.apply(turn_on_dropout)  # 在 eval 也能开启 dropout，用于 AL


def train_eval_dropout():
    """
    dropout 采用 bagging 思想
        bagging:
            使用 bootstrap-sample 在不同数据集上 训练得到不同模型，最后 sum 各个模型的结果作为最终结果，各个模型独立
            降低方差，降低过拟合；针对强学习器，如决策树
        boosting:
            包括 Adaboost 和 GBDT
            串行集成一系列弱学习器，模型是有顺序的; 根据上一个模型的预测结果，reweight sample 发现困难样本

    train/test 时 scale 是为了保障 输入输出分布的期望一致
    Inverted dropout 在 train 时 scale dropout 后的结果，test 时就不用改变了
    为什么是保证 x_after_dropout 和 x_before_dropout 的期望相同 ?
    因为 dropout 层的输出 作为下一层的输入，来学习每个 w; 而每个 w 的输入是 w*x 求和的形式，将 x 放大，w 就不会学大了

    最初版本的 dropout 在 train 上没有 放大输出，而在 test 时缩小输出
        放大输出后，train 时的 w 就不会学得偏大了
    """
    a = torch.ones(10)
    print(a)

    m = nn.Dropout(p=0.2)
    print(m.training)  # True; 默认 train 状态
    print(m(a))  # scale output 输出, * 1 / (1-p)

    m.eval()
    print(m(a))  # 全部 神经元 使用


def diff_drop_drop2d():
    a = torch.ones(1, 2, 4, 4) * 10
    r = 0.5
    print(a)
    print(nn.Dropout(r)(a))  # 每张 feature map 随机去掉 一半 pixel
    print(nn.Dropout2d(r)(a))  # 将 一半的 feature maps 直接置为 0


if __name__ == '__main__':
    diff_drop_drop2d()
