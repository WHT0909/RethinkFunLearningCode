import torch
from torch.utils.tensorboard import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"
# 三个参数：w0 w1 w2
w_real = torch.tensor([1.1, 2.2, 3.3], requires_grad=False).reshape((3, -1)).to(device) # 真实参数无需更新，不需要求导
# 偏置
b_real = torch.tensor(4.4, requires_grad=False).to(device)

# 生成模拟数据
inputs = torch.randn((100, 3))
outputs = inputs @ w_real + b_real
outputs +=  torch.randn(outputs.shape, device=device)

# 定义超参数
lr = 0.003
epochs = 3000
writer = SummaryWriter(log_dir="D:\Github_Projects\RethinkFunLearningCode\Chapter2_初识Pytorch\loss_curve")

# 初始化参数
w = torch.rand((3, 1), requires_grad=True, device=device)
b = torch.rand((1, ), requires_grad=True, device=device)

inputs.to(device)
outputs.to(device)

for epoch in range(epochs):
    pred = inputs @ w + b
    loss = torch.mean(torch.square(pred - outputs))
    if (epoch + 1) % 100 == 0:
        print(f"第{epoch+1}轮 loss：{loss.item()}")
    writer.add_scalar("loss/train", loss.item(), epoch)
    loss.backward()
    with torch.no_grad(): # 不影响计算图
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_()
    b.grad.zero_()

print(f"计算结果：w:{w}\tb:{b}")