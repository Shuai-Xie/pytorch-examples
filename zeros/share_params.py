"""
pytorch 如何共享参数
https://www.cnblogs.com/sdu20112013/p/12134330.html
"""
import torch
import torch.nn as nn
import torch.nn.init as init


def seq_share_weights():
    linear = nn.Linear(1, 1, bias=False)
    # 传入 Sequential 的模块是同一个 Module 实例的话参数也是共享的
    net = nn.Sequential(linear, linear)  # 2个 linear 在内存中对应同一个对象
    print(net)
    # id -> 139669020411160，对象 id，unique among simultaneously existing objects.
    print(id(net[0]) == id(net[1]))  # True
    print(id(net[0].weight) == id(net[1].weight))  # True

    # y = wx, 初始化 linear 层 w=3; net = 3*3*x = 9x
    for name, param in net.named_parameters():
        init.constant_(param, val=3)
        print(name, param.data)

    x = torch.ones(1, 1)  # bs=1
    y = net(x).sum()
    print(y)  # 3*3*1 = 9
    y.backward()
    print(net[1].weight.grad)  # 6 共享参数的 grad 是累加的，相当于更新了2次
    print(net[0].weight.grad)  # 6


def seq_unique_weights():
    linear1 = nn.Linear(1, 1, bias=False)
    linear2 = nn.Linear(1, 1, bias=False)
    net = nn.Sequential(linear1, linear2)
    print(net)

    for name, param in net.named_parameters():
        init.constant_(param, val=3)
        print(name, param.data)

    x = torch.ones(1, 1)
    y = net(x).sum()
    print(y)
    y.backward()
    print(net[1].weight.grad)  # 3; 倒数第1层，grad1 = 3x = 3
    print(net[0].weight.grad)  # 3; 倒数第2层，grad0 = grad1 * x = 3*1 = 3


if __name__ == '__main__':
    seq_share_weights()
    print()
    seq_unique_weights()
