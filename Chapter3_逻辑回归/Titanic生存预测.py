import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

class TitanicDataset(Dataset):
    """定义 Titanic 数据集类"""
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self._load_data()
        self.feature_size = len(self.data.drop(columns=["Survived"]).columns)

    def _load_data(self):
        """数据预处理"""
        df = pd.read_csv(self.file_path)
        df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
        df = df.dropna(subset=["Age"])
        df = pd.get_dummies(df, columns=["Sex", "Embarked"], dtype=int)
        # 标准化
        base_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
        for i in range(len(base_features)):
            df[base_features[i]] = (df[base_features[i]] - df[base_features[i]].mean()) / df[base_features[i]].std()
        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        features = self.data.drop(columns=["Survived"]).iloc[item].values
        labels = self.data["Survived"].iloc[item]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

class LogisticRegressionModel(nn.Module):
    """定义逻辑回归模型"""
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

train_dataset = TitanicDataset("titanic_data/train_data.csv")
val_dataset = TitanicDataset("titanic_data/val_data.csv")
lr = 1e-3
epochs = 100

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LogisticRegressionModel(input_dim=train_dataset.feature_size).to(device)
model.train()
optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
for epoch in range(epochs):
    correct = 0
    step = 0
    total_loss = 0
    for features, labels in DataLoader(train_dataset, batch_size=256):
        step += 1
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(features).squeeze()
        correct += torch.sum((outputs > 0.5) == labels)
        loss = torch.nn.functional.binary_cross_entropy(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch+1}\tLoss:{total_loss / step:.4f}")
    print(f"Train ACC:{correct / len(train_dataset)}")
model.eval()
with torch.no_grad():
    correct = 0
    for features, labels in DataLoader(val_dataset, batch_size=256):
        features = features.to(device)
        labels = labels.to(device)
        outputs = model(features).squeeze()
        correct += torch.sum((outputs > 0.5) == labels)
    print(f"Test ACC:{correct / len(val_dataset)}")