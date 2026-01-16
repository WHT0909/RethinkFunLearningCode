# 利用 pytorch 自动计算梯度

# 要计算的式子：\[ z = \log{(3x+4y)^{2}} \]
import torch

x = torch.tensor(1., requires_grad=True)
y = torch.tensor(1., requires_grad=True)
v = 3 * x + 4 * y
u = torch.square(v)
z = torch.log(u)

z.backward()

print(x.grad)
print(y.grad)