import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
X = torch.tensor([[2, 1000], [3, 2000], [2, 500], [1, 800], [4, 3000]], device=device, dtype=torch.float) # (5,2)
y = torch.tensor([[19], [31], [14], [15], [43]], device=device, dtype=torch.float) # (5,1)
# print(X.shape, y.shape)

# 标准化，防止 loss 过大
X_mean = X.mean(dim=0) # dim=0 可以理解为消灭“行”，也就是把矩阵拍扁让他就剩一行
X_std = X.std(dim=0)
X = (X - X_mean) / X_std

w = torch.rand((2, 1), device=device, requires_grad=True)
b = torch.rand((1, ), device=device, requires_grad=True)

lr = 0.01
epochs = 5000

for epoch in range(epochs):
    y_pred = X @ w + b
    loss = torch.mean(torch.square(y - y_pred))
    if (epoch + 1) % 100 == 0:
        print(f"第{epoch + 1}轮\tloss:{loss.item():.5f}")
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
print(f"w:{w}\tn:{b}")