import numpy as np


def my_cov(a):
    mean = a.mean(axis=1)  # 沿列
    x = a.T - mean  # n,dim
    return x.T.dot(x) / (x.shape[0] - 1)  # 无偏估计协方差 /n-1


n, dim = 10, 3
a = np.random.randn(dim, n)  # 注意每一维样本 要以 列向量形式表示
print(np.cov(a))
print(my_cov(a))
