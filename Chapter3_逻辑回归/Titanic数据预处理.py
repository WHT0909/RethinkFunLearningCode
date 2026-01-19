import pandas as pd

df = pd.read_csv("./titanic_data/train.csv")
#去除不需要的列
df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
# 去除 Age 缺失的样本
df.dropna(subset=["Age"])
# 对 Sex 和 Embarked 做独热编码
df = pd.get_dummies(df, columns=["Sex", "Embarked"], dtype=int)
print(df.head(10))