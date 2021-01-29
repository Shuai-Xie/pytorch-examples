import torch

"""
torch.no_grad() impacts the autograd engine and deactivate it. 关闭 autograd 加快计算 
It will reduce memory usage and speed up. 

model.eval() will notify all your layers that you are in eval mode, that way, 固定 dropout/bn 等的参数 
batchnorm or dropout layers will work in eval mode instead of training mode.

dropout 训练时 随机丢弃
batchnorm 训练时 更新 mean,std; 测试时使用 所有批的统计结果

---

叶节点的 grad 会保留；而非叶节点的 grad 会在反传 w = w - lr*grad 更新后，就释放掉
可以用 registe_hook 函数获取中间层参数，包括: register_forward_hook / register_backward_hook
"""


def demo():
    """
    非叶子节点，即计算得到的中间变量
    在 loss backward 之前，inplace 修改会报错；因为求偏导时，可能需要用到其在 version=0 的值；而 inplace 会使这一值丢失
    """
    a = torch.tensor([1.0, 3.0], requires_grad=True)
    b = a + 2
    loss = (b * b).mean()
    print(a._version)  # 0
    print(b._version)  # 0
    print(loss._version)  # 0

    b[0] = 1000.0  # 1，inplace 操作
    # b += 2  # 1，inplace 操作
    # b = b + 2  # 0, 非 inplace
    print(b._version)  # 1

    loss.backward()
    # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation:
    # [torch.FloatTensor [2]], which is output 0 of AddBackward0, is at version 1; expected version 0 instead
    # 可以看到 已经检测出 b 与 loss 的 version 不对应了;
    # 每次 tensor 做 inplace 操作, 变量 _version 就会加1，其初始值为0


def demo_leaf1():
    """
    叶子节点; loss backward 之前 把输入给修改了...
    """
    a = torch.tensor([10., 5., 2., 3.], requires_grad=True)
    print(a, a.is_leaf)
    # tensor([10.,  5.,  2.,  3.], requires_grad=True) True

    # inplace 修改输入 a 之后，输入变成了中间变量；这时原来的输入就没了
    a[:] = 0
    print(a, a.is_leaf)
    # tensor([0., 0., 0., 0.], grad_fn=<CopySlices>) False

    loss = (a * a).mean()
    loss.backward()
    # RuntimeError: leaf variable has been moved into the graph interior 叶节点移进图内部


def demo_leaf2():
    a = torch.tensor([10., 5., 2., 3.], requires_grad=True)
    a.add_(10.)
    # RuntimeError: a leaf Variable that requires grad has been used in an in-place operation.
    # 需要求导的值，inplace 修改后，梯度更新函数的输入没了...


def howto_inplace_change_leaf():
    """
    如何 inplace 不改内存的 修改 leaf 的值
    """
    # 方法一
    a = torch.tensor([10., 5., 2., 3.], requires_grad=True)
    print(a, a.is_leaf, id(a))
    # tensor([10.,  5.,  2.,  3.], requires_grad=True) True 2501274822696

    # 改变 leaf 节点值，而不改变地址位置，并且没有将 a 移到计算图内部
    # print(a.data)  # tensor([10.,  5.,  2.,  3.]) 修改共享内存的不在图中的 data 的值; 而不是直接修改图中的 a
    a.data.fill_(10.)
    # 或者 a.detach().fill_(10.)
    print(a, a.is_leaf, id(a))
    # tensor([10., 10., 10., 10.], requires_grad=True) True 2501274822696

    loss = (a * a).mean()
    loss.backward()
    print(a.grad)
    # tensor([5., 5., 5., 5.])

    # 方法二
    a = torch.tensor([10., 5., 2., 3.], requires_grad=True)
    print(a, a.is_leaf)
    # tensor([10.,  5.,  2.,  3.], requires_grad=True) True

    with torch.no_grad():  # 把 autograd 关掉，自然就不会报错了；而不是修改所有 tensor 的 require_grad 属性
        a[:] = 10.
        print(a)
    print(a, a.is_leaf)
    # tensor([10., 10., 10., 10.], requires_grad=True) True

    loss = (a * a).mean()
    loss.backward()
    print(a.grad)
    # tensor([5., 5., 5., 5.])


howto_inplace_change_leaf()
