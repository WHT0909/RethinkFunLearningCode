import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

class MyTitanicDataset(Dataset):
    """自定义数据类"""
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = self._load_data()
        self.feature_size = len(self.data.drop(columns=["Survived"]).columns)
    def _load_data(self):
        data = pd.read_csv(self.data_path)
        data = data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
        data = data.dropna(subset=["Age"])
        data = pd.get_dummies(data, columns=["Sex", "Embarked"], dtype=int)
        base_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
        for col in base_features:
            data[col] = (data[col] - data[col].mean()) / data[col].std()
        return data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        labels = self.data["Survived"].iloc[item]
        features = self.data.drop(columns=["Survived"]).iloc[item].values
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

class MyLinearRegressionModel(nn.Module):
    """自定义逻辑回归模型"""
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, out_features=1)
    def forward(self, X):
        return torch.sigmoid(self.linear(X))

train_dataset = MyTitanicDataset("./titanic_data/train_data.csv")
val_dataset = MyTitanicDataset("./titanic_data/val_data.csv")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MyLinearRegressionModel(input_dim=train_dataset.feature_size).to(device)
writer = SummaryWriter(log_dir="D:\\Github_Projects\\RethinkFunLearningCode\\Chapter3_逻辑回归\\titanic_loss_curve")
epochs = 500
lr = 1e-3
optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
model.train()
for epoch in range(epochs):
    total_loss = 0
    step = 0
    train_correct = 0
    for features, labels in DataLoader(train_dataset, batch_size=128, shuffle=True):
        step += 1
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(features).squeeze()
        train_correct += torch.sum((outputs > 0.5) == labels)
        loss = torch.nn.functional.binary_cross_entropy(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = total_loss / step
    writer.add_scalar("train_loss", train_loss, epoch)
    print(f"Epoch {epoch+1} loss:{train_loss:.4f}")
    train_acc = train_correct / len(train_dataset)
    writer.add_scalar("train_acc", train_acc, epoch)
    print(f"Train ACC:{train_acc}")


model.eval()
with torch.no_grad():
    val_correct = 0
    val_num_samples = 0
    for features, labels in DataLoader(val_dataset, batch_size=256, shuffle=True):
        outputs = model(features).squeeze()
        val_correct += torch.sum((outputs > 0.5) == labels)
        val_num_samples += len(features)
    print(f"Test ACC:{val_correct / val_num_samples}")