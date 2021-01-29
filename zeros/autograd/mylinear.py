import torch
import torch.nn as nn
from torch.nn import init

from torch.autograd.function import Function, once_differentiable  # 针对某些不可求导的操作，返回一个结果


class MyLinear(Function):
    """
    y = w * x + b
        grad_x = grad_output * w
        grad_w = grad_output * x
        grad_b = grad_output * 1

    正向 y = x * w.T
    反向 x = y * w
    只是 shape 满足，并不是真实值
    """

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)  # 将Tensor转变为Variable保存到ctx中
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)  # N,C
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # print(ctx.needs_input_grad)  # (True, True, True)

        if ctx.needs_input_grad[0]:  # input
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)  # N,C 对 N 求和

        return grad_input, grad_weight, grad_bias


# 为 apply 取别名，方便直接 forward
_linear = MyLinear.apply


# 扩展自定义 func 为 module
class MyLinearModule(nn.Module):

    def __init__(self, input_features, output_features, bias=True):
        super(MyLinearModule, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # 定义训练参数 | Parameter 默认 require_grad=True
        # torch.Tensor 默认初始化为 全 -> 0
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.bias = bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        self._init_weights()

    def _init_weights(self):
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.zero_()  # 0 init

    def forward(self, input):
        return MyLinear.apply(input, self.weight, self.bias)


def check_grad():
    from torch.autograd import gradcheck
    inputs = torch.randn((1, 10), requires_grad=True).double()  # B,C
    linear = MyLinearModule(input_features=10, output_features=3).double()
    test = gradcheck(lambda x: linear(x), (inputs,))
    print(test)

    # https://discuss.pytorch.org/t/why-does-gradcheck-fail-for-floats/31387
    # 32 bit floating point usually doesn’t have enough precision to pass numerical gradient checks.??

    # double 模式检查 不会 user warning; Jacobian mismatch 偏导矩阵不匹配
    # inputs = torch.randn((1, 1, 2, 2), requires_grad=True).double()
    # conv = nn.Conv2d(1, 1, 1, 1).double()
    # test = gradcheck(lambda x: conv(x), (inputs,))
    # print(test)


if __name__ == '__main__':
    check_grad()
    pass
