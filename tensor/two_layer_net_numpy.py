import numpy as np

"""
A fully-connected ReLU network with one hidden layer and no biases, trained to
predict y from x using Euclidean error.

This implementation uses numpy to manually compute the forward pass, loss, and
backward pass.

A numpy array is a generic n-dimensional array; it does not know anything about
deep learning or gradients or computational graphs, and is just a way to perform
generic numeric computations.
"""

# N is batch size;
# D_in is input dimension;
# H is hidden dimension;
# D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6

for t in range(500):
    # Forward pass: compute predicted y
    h = x.dot(w1)  # hidden: N,H
    h_relu = np.maximum(h, 0)  # relu, N,H
    y_pred = h_relu.dot(w2)  # output: N,D_out

    # Compute and print loss
    # loss 函数，求 batch 的 sum or mean 作为反向传播起点
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    # y = w2 * [ relu(w1 * x) ]
    # loss
    # 对 y_pred 求导, 得到 y_pred 的梯度，即输入的 y_pred 下次更新需要下降的值！
    grad_y_pred = 2.0 * (y_pred - y)
    # y_pred = h_relu @ w2
    # 对 w2 求导，链式相乘, 注意可通过 size 一直判断是否转置
    grad_w2 = h_relu.T.dot(grad_y_pred)  # H,N * N,D_out -> H,D_out
    # 对 relu 求导
    grad_h_relu = grad_y_pred.dot(w2.T)  # N,D_out * D_out,H -> N,H
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    # 对 w1 求导
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
