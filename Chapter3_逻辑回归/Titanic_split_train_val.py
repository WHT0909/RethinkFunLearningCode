import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

def split_train_val_data(input_data_path, train_output_path, val_output_path, test_size=0.2, random_state=42):
    """分割训练集和验证集，并把训练集和验证集写到 csv 文件中"""
    df = pd.read_csv(input_data_path)
    df_train, df_val = train_test_split(df, test_size=test_size, random_state=random_state)
    df_train.to_csv(train_output_path, index=False, header=True)
    df_val.to_csv(val_output_path, index=False, header=True)
    print("已完成数据集分割")


if __name__ == "__main__":
    data_path = "./titanic_data/train.csv"
    split_train_val_data(data_path, "./titanic_data/train_data.csv", "./titanic_data/val_data.csv")
