import numpy as np


class FCNet:
    def __init__(self, B, D_in, D_hid, D_out):
        # FC 网络 两层参数
        self.w1 = np.random.randn(D_in, D_hid)
        self.w2 = np.random.randn(D_hid, D_out)

        # make data, 常量
        self.X = np.random.randn(B, D_in)  # 指定数据输入维度
        self.y = np.random.randn(B, D_out)

        # lr
        self.lr = 1e-6

    def forward_and_backwards(self):
        # forward
        # y = w2 * (relu(w1 * X))  # () 表示层的嵌套
        h = self.X @ self.w1  # B,D_hid
        h_relu = np.maximum(h, 0)  # 作为 layer2 输入，使用 np 方便 @ 乘
        y_pred = h_relu @ self.w2  # B,D_out

        # MSE loss = (y_pred - y)^2
        loss = np.square(y_pred - self.y).sum()
        print(loss)

        # grad
        # y_pred, h_relu, h 这些都可认为是中间 feature 值
        # 目标是对 w1,w2 求导，不方便直接求导，所以先求 y_pred 再链式传过去
        grad_y_pred = 2 * (y_pred - self.y)  # B,D_out
        # w2 偏导，形式 f = a * w2
        grad_w2 = h_relu.T @ grad_y_pred  # D_hid,D_out
        # 为了求 w1 偏导，需要链式求 h_relu 导数
        grad_h_relu = grad_y_pred @ self.w2.T  # B,D_hid
        # 对 relu 求导
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0  # note: 是 h<0 对应的 dim 的 grad=0，而不是 grad<0
        # >=0 相当于偏导 * 1 还是原值
        # 梯度再传给 w1
        grad_w1 = self.X.T @ grad_h

        # backward
        self.w1 -= self.lr * grad_w1
        self.w2 -= self.lr * grad_w2

    def train(self, steps=500):
        for i in range(steps):
            print(i, end=': ')
            self.forward_and_backwards()


if __name__ == '__main__':
    model = FCNet(64, 1000, 100, 10)
    model.train(steps=500)
