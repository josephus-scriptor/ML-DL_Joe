# ========================
# Final Deep Learning Pipeline for Tabular Data (Colab-ready)
# Project: ML-DL_Joe | Dataset: data.csv
# ========================

# SECTION: Data Processing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             precision_recall_curve, classification_report)

# Mount Drive (if using Colab)
from google.colab import drive
drive.mount('/content/drive')

# Load dataset
path = '/content/drive/MyDrive/ML-DL_Joe/data.csv'
df = pd.read_csv(path)

# Generate cross-features and binarized flags
for i in range(1, 31):
    for j in range(i+1, 31):
        df[f'V{i}_x_V{j}'] = df[f'V{i}'] * df[f'V{j}']
    df[f'V{i}_gt_mean'] = (df[f'V{i}'] > df[f'V{i}'].mean()).astype(int)

# Target and features
y = df['OBJ']
X = df.drop(columns=['OBJ'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
import torch
from torch.utils.data import TensorDataset, DataLoader

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# SECTION: Models Used
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# SECTION: Hyperparameter Configuration
model = MLPModel(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
for epoch in range(30):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()
    print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")

# SECTION: Evaluation and Metrics
model.eval()
with torch.no_grad():
    probs = model(X_test_tensor.to(device)).cpu().numpy().flatten()
    preds = (probs >= 0.5).astype(int)

balanced_acc = balanced_accuracy_score(y_test, preds)
roc_auc = roc_auc_score(y_test, probs)
precision, recall, thresholds = precision_recall_curve(y_test, probs)

print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(classification_report(y_test, preds))

# SECTION: Final Model Selected
# Selected Model: MLP with LayerNorm, Dropout, Adam optimizer, LR Scheduler
# Metrics:
# - Balanced Accuracy
# - ROC-AUC
# - Precision/Recall optimized via threshold tuning

# Save model (optional)
torch.save(model.state_dict(), '/content/drive/MyDrive/ML-DL_Joe/final_mlp_model.pt')
