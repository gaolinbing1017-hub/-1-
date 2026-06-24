import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import warnings
import sys

# 导入 GUI 弹窗相关库
import tkinter as tk
from tkinter import filedialog

warnings.filterwarnings('ignore')

# ==========================================
# 0. 弹出窗口选择数据集文件
# ==========================================
# 创建一个隐藏的主窗口
root = tk.Tk()
root.withdraw() 
# 将窗口置顶，防止被其他窗口挡住
root.attributes('-topmost', True) 

print("正在等待选择数据集文件，请在弹出的窗口中选择...")
# 弹出文件选择对话框
file_path = filedialog.askopenfilename(
    title="请选择心跳信号分类数据集 (CSV格式)",
    filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")]
)

# 检查用户是否选择了文件，如果点了取消则退出程序
if not file_path:
    print("❌ 未选择任何文件，程序已退出。")
    sys.exit()

print(f"✅ 已成功选择文件:\n {file_path}\n")

# ==========================================
# 1. 数据加载与预处理
# ==========================================
print("正在加载和处理数据...")
# 使用用户选择的路径读取数据
df = pd.read_csv(file_path)

# 解析 heartbeat_signals，将其从逗号分隔的字符串转换为 float 类型的 numpy 数组
df['heartbeat_signals'] = df['heartbeat_signals'].apply(lambda x: np.array([float(i) for i in x.split(',')]))

# 构建特征 X 和标签 y
X = np.stack(df['heartbeat_signals'].values)
y = df['label'].values

# 按照 8:2 划分训练集和测试集，使用 stratify=y 保证划分后类别比例一致
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"数据划分完成 -> 训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# 计算类别权重 (应对类别数据极度不平衡的问题)
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = torch.tensor(weights, dtype=torch.float32)
print(f"计算得到的类别权重: {class_weights.numpy()}\n")

# 自定义 Dataset
class HeartbeatDataset(Dataset):
    def __init__(self, signals, labels):
        # 增加一个维度代表 channel (1D 卷积需要输入维度为 [Batch, Channel, Length])
        self.signals = torch.tensor(signals, dtype=torch.float32).unsqueeze(1) 
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]

# 创建 DataLoader
batch_size = 128
train_dataset = HeartbeatDataset(X_train, y_train)
test_dataset = HeartbeatDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==========================================
# 2. 模型定义 (1D CNN)
# ==========================================
class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        # 特征提取层：3层 1D卷积 + BN + ReLU + MaxPool
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        # 分类层
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), # 防止过拟合
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4) # 4 个类别
        )

    def forward(self, x):
        x = self.features(x)
        # 全局平均池化 (Global Average Pooling) -> [Batch, Channels, Length] 变成 [Batch, Channels]
        x = torch.mean(x, dim=2) 
        x = self.classifier(x)
        return x

# 配置设备 (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用的计算设备: {device}\n")

model = CNN1D().to(device)
class_weights = class_weights.to(device)

# ==========================================
# 3. 定义损失函数、优化器和学习率调度器
# ==========================================
# 传入类别权重应对不平衡
criterion = nn.CrossEntropyLoss(weight=class_weights) 
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# ==========================================
# 4. 模型训练与验证
# ==========================================
epochs = 50
print("🚀 开始训练模型...")

for epoch in range(epochs):
    # --- 训练阶段 ---
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad() # 梯度清零
        outputs = model(inputs) # 前向传播
        loss = criterion(outputs, labels) # 计算损失
        loss.backward() # 反向传播
        optimizer.step() # 更新参数
        
        train_loss += loss.item() * inputs.size(0)
    
    scheduler.step() # 更新学习率
    train_loss /= len(train_loader.dataset)
    
    # --- 每 5 个 Epoch 打印一次训练进度 ---
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f'Epoch [{epoch+1:02d}/{epochs}], 训练集 Loss: {train_loss:.4f}, 当前学习率: {scheduler.get_last_lr()[0]:.6f}')

# ==========================================
# 5. 模型测试与评价指标计算
# ==========================================
print("\n正在对测试集进行评估...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1) # 获取最大概率的索引作为预测类别
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 计算评价指标
acc = accuracy_score(all_labels, all_preds)
# Macro F1-Score 针对类别不平衡更具参考意义
f1 = f1_score(all_labels, all_preds, average='macro') 

print("\n================ 测试结果 ================")
print(f"总体准确率 (Accuracy) : {acc * 100:.2f}%")
print(f"宏平均 F1-Score (Macro F1) : {f1:.4f}")
print("==========================================")