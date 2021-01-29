import torch

x = torch.randn(10)

eps = 1e-5
print(x.sigmoid().clamp(min=eps, max=1 - eps))

print(x.mean())
print(x.std())

import math

print(math.log(999))

_liner = 'haha'
