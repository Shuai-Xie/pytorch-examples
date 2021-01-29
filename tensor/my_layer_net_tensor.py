import torch


class FCNet:
    def __init__(self, B, D_in, D_hid, D_out):
        # FC 网络 两层参数
        self.w1 = torch.randn((D_in, D_hid))
        self.w2 = torch.randn((D_hid, D_out))

        # make data, 常量
        self.X = torch.randn(B, D_in)  # 指定数据输入维度
        self.y = torch.randn(B, D_out)  # target 指定维度

        # lr
        self.lr = 1e-6

    # l1 拟合更慢
    def loss_fn_grad(self, y_pred, y_target, mode):
        if mode == 'l1':
            loss = torch.abs(y_pred - y_target).sum()
            grad_y_pred = torch.sign(y_pred - y_target)  # 符号函数分段求导, >0 1; <0 -1; =0 0
        elif mode == 'l2':
            loss = (y_pred - y_target).pow(2).sum()
            grad_y_pred = 2 * (y_pred - y_target)
        else:
            raise NotImplementedError
        return loss, grad_y_pred

    def forward_and_backwards(self):
        # forward
        # y = w2 * (relu(w1 * X))  # () 表示层的嵌套
        h = self.X.mm(self.w1)
        h_relu = h.clamp(min=0)  # relu
        y_pred = h_relu.mm(self.w2)

        # l1 loss = |y_pred - y|
        loss, grad_y_pred = self.loss_fn_grad(y_pred, self.y, mode='l1')
        # loss, grad_y_pred = self.loss_fn_grad(y_pred, self.y, mode='l2')
        print(loss)

        # grad
        # w2 偏导，形式 f = a * w2
        grad_w2 = h_relu.t().mm(grad_y_pred)  # D_hid,D_out
        # 为了求 w1 偏导，需要链式求 h_relu 导数
        grad_h_relu = grad_y_pred.mm(self.w2.t())
        # 对 relu 求导
        grad_h = grad_h_relu.clone()
        grad_h[h < 0] = 0  # note: 是 h<0 对应的 dim 的 grad=0，而不是 grad<0
        # >=0 相当于偏导 * 1 还是原值
        # 梯度再传给 w1
        grad_w1 = self.X.t().mm(grad_h)

        # backward
        self.w1 -= self.lr * grad_w1
        self.w2 -= self.lr * grad_w2

    def train(self, steps=500):
        for i in range(steps):
            print(i, end=': ')
            self.forward_and_backwards()


if __name__ == '__main__':
    model = FCNet(64, 1000, 100, 10)
    model.train(steps=5000)
