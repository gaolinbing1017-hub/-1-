import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import warnings
import sys
import tkinter as tk
from tkinter import filedialog

# 导入 HuggingFace 库中构建 Bert 模型需要的模块
from transformers import BertConfig, BertForSequenceClassification, get_linear_schedule_with_warmup

warnings.filterwarnings('ignore')

# ==========================================
# 0. 弹出窗口选择数据集文件
# ==========================================
# 初始化隐藏的 tkinter 主窗口
root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True) # 窗口置顶

print("正在等待选择数据集文件，请在弹出的窗口中选择...")
# 弹出对话框
file_path = filedialog.askopenfilename(
    title="请选择新闻文本分类数据集 (CSV格式)",
    filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")]
)

if not file_path:
    print("❌ 未选择任何文件，程序已退出。")
    sys.exit()

print(f"✅ 已成功选择文件:\n {file_path}\n")

# ==========================================
# 1. 数据加载与预处理
# ==========================================
print("正在加载和处理数据...")
# 天池新闻数据集默认由 \t 分割，务必使用 sep='\t'
df = pd.read_csv(file_path, sep='\t')

# 提取特征和标签
X = df['text'].values
y = df['label'].values

# 由于原比赛数据量很大（约20万条），如果你的显卡显存不够，
# 可以截取部分数据运行，例如去除下面的注释：
# X = X[:20000]
# y = y[:20000]

print("正在划分训练集和测试集 (比例 8:2)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"数据划分完成 -> 训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

# 自动计算类别数量
num_classes = len(np.unique(y))
print(f"共发现 {num_classes} 个新闻类别。")

# ==========================================
# 2. 自定义 Dataset 和 动态填充 Collate 函数
# ==========================================
class NewsDataset(Dataset):
    def __init__(self, texts, labels, max_len=256):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 比赛数据是由空格分隔的整数 ID 字符串
        # 按照 max_len 截断超长的新闻文本
        text_str = str(self.texts[idx])
        token_ids = [int(x) for x in text_str.split()][:self.max_len]
        
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def collate_fn(batch):
    texts, labels = zip(*batch)
    # 动态将一个 batch 内的长度填充至最长序列的长度，0 为 padding_value
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    
    # 构建 Attention Mask，告诉多头注意力机制哪些位置是补零的，不需要计算注意力权重
    attention_mask = (texts_padded != 0).long()
    
    labels = torch.stack(labels)
    return texts_padded, attention_mask, labels

# 创建 DataLoader
batch_size = 32  # 如果使用 Bert 导致 CUDA Out of Memory，请调小至 16 或 8
max_seq_len = 256 # 文本截断长度

train_dataset = NewsDataset(X_train, y_train, max_len=max_seq_len)
test_dataset = NewsDataset(X_test, y_test, max_len=max_seq_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# ==========================================
# 3. 模型定义 (基于 BERT 架构)
# ==========================================
print("\n正在构建 BERT 模型...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用的计算设备: {device}\n")

# 动态计算最大词表大小 (Vocab Size) 以初始化 Embedding 层
all_words = []
for text in X_train:
    all_words.extend([int(word) for word in str(text).split()])
vocab_size = max(all_words) + 1
print(f"从训练集提取的 Vocab Size: {vocab_size}")

# 我们不使用预训练权重，而是手动配置一个包含多头注意力机制的轻量级 Bert
config = BertConfig(
    vocab_size=vocab_size,
    hidden_size=256,       # Transformer 的隐藏层维度
    num_hidden_layers=4,   # 包含 4 层 Transformer 编码器
    num_attention_heads=4, # 设定 4 个多头注意力 (Multi-Head Attention)
    intermediate_size=1024,
    max_position_embeddings=512,
    num_labels=num_classes # 输出层类别数
)

# 使用自定义配置初始化分类模型
model = BertForSequenceClassification(config).to(device)

# ==========================================
# 4. 定义优化器、学习率预热机制与损失函数
# ==========================================
epochs = 5
learning_rate = 2e-4

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Transformer 模型训练常需要学习率 Warmup 机制
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps
)

# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# ==========================================
# 5. 模型训练与验证
# ==========================================
print("🚀 开始训练模型...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for step, (input_ids, attention_mask, labels) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad() # 清除历史梯度
        
        # 将输入送入模型。Huggingface 的 BertForSequenceClassification 已经封装了 Loss 的计算
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward() # 反向传播
        
        # 梯度裁剪防爆炸 (Transformer 标准操作)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step() # 更新学习率
        
        total_loss += loss.item()
        
        # 打印当前进度
        if (step + 1) % 100 == 0 or (step + 1) == len(train_loader):
            print(f"Epoch [{epoch+1}/{epochs}], Step [{step+1}/{len(train_loader)}], 当前批次 Loss: {loss.item():.4f}")
            
    avg_train_loss = total_loss / len(train_loader)
    print(f"==> Epoch {epoch+1} 结束, 平均训练 Loss: {avg_train_loss:.4f}\n")

# ==========================================
# 6. 模型测试与评价
# ==========================================
print("正在对测试集进行评估...")
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for input_ids, attention_mask, labels in test_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1) # 获取预测得分最高的一类
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.numpy())

# 计算评价指标
acc = accuracy_score(all_targets, all_preds)
f1 = f1_score(all_targets, all_preds, average='macro')

print("\n================ 测试结果 ================")
print(f"总体准确率 (Accuracy) : {acc * 100:.2f}%")
print(f"宏平均 F1-Score (Macro F1) : {f1:.4f}")
print("==========================================")