import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import LabelBinarizer
import numpy as np

# 加载灰度数据集
train_data = pd.read_csv("train_data_grey.csv")
val_data = pd.read_csv("val_data_grey.csv")
test_data = pd.read_csv("test_data_grey.csv")

# 提取特征和标签
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

X_val = val_data.iloc[:, :-1].values
y_val = val_data.iloc[:, -1].values

X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# 转换标签为整数索引
y_train_tensor = torch.tensor(y_train.astype(int), dtype=torch.long)
y_val_tensor = torch.tensor(y_val.astype(int), dtype=torch.long)
y_test_tensor = torch.tensor(y_test.astype(int), dtype=torch.long)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)


# 构造数据加载器
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout_rate, use_bn):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        prev_dim = input_dim

        # 构建隐藏层
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_bn:
                self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.ReLU())
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(dropout_rate))  # Dropout
            prev_dim = hidden_dim

        # 输出层
        self.layers.append(nn.Linear(prev_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def create_mlp(input_dim, output_dim, hidden_dims, dropout_rate, use_bn):
    if hidden_dims is None:
        hidden_dims = [2048, 1024, 512, 256, 64]
    return MLP(input_dim, output_dim, hidden_dims,dropout_rate, use_bn)

# 初始化模型
input_dim = X_train.shape[1]  # 输入维度
output_dim = len(torch.unique(y_train_tensor))  # 输出类别数
hidden_dims = [1024,512,256,64]  # 隐藏层结构
model = create_mlp(input_dim, output_dim, hidden_dims,dropout_rate=0.4, use_bn=True)

# 检查是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)


# 定义加权损失函数
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5,weight_decay=1e-4)

# 训练模型
epochs = 20
train_loss = []
val_total_loss = []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, labels)

        # 梯度清零
        optimizer.zero_grad()

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    loss = running_loss / len(train_loader)
    train_loss.append(loss)

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {running_loss / len(train_loader):.4f}")

    # 验证阶段
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss = val_loss / len(val_loader)
    val_total_loss.append(loss)
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {100 * correct / total:.2f}%")

import matplotlib.pyplot as plt

# 绘制训练和验证损失
plt.plot(train_loss, label="Train Loss")
plt.plot(val_total_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train and Validation Loss")
plt.legend()
plt.show()

# 测试模型
model.eval()
test_loss = 0.0
correct = 0
total = 0


with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        # 计算准确率
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {100 * correct / total:.2f}%")
