import torch

"""
Einstein summation convention 爱因斯坦求和约定
"""
# a = torch.arange(6).reshape(2, 3)
# b = torch.einsum('ij->ji', [a])
# print(a)
# print(b)


"""
flip
"""

# x = torch.arange(8).view(2, 2, 2)
# print(x)
# x = x.flip(-1)  # 必须返回值，不会 inpalce 修改
# print(x)

"""
gather
Gathers values along an axis specified by dim.

for a 3D tensor
out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

note: 
- index 和 input 的 dim 必须相同，比如 dim 都 =2
- gather 输出结果 shape 和 index 一致

index 要制定 gather 指定 dim 的 index 的值
"""

a = torch.arange(1, 21).view(4, 5)
print(a)
index = torch.randint(0, 4, size=a.size())
print(index)
res = a.gather(dim=0, index=index)  # dim=0 表示沿行取 取出的数 形成 1,4
print(res)
res = a.gather(dim=1, index=index)
print(res)

"""
init
"""
# a = torch.randn((2, 3))
# print(a.uniform_(0, 1))  # inplace 修改
# print(a.normal_(mean=0, std=1))
