import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Dataset
# -----------------------------
class MyDataset(Dataset):
    def __init__(self):
        # random regression data
        self.x = torch.randn(500, 10)
        self.y = torch.randn(500, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


dataset = MyDataset()

# train/validation split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)


# -----------------------------
# Model
# -----------------------------
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.act = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x


model = SimpleNN()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# Training
# -----------------------------
epochs = 20
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    running_train_loss = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # validation loop
    model.eval()
    running_val_loss = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            running_val_loss += loss.item()

    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")


# -----------------------------
# Evaluation Metrics
# -----------------------------
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for xb, yb in val_loader:
        pred = model(xb)
        all_preds.append(pred.numpy())
        all_targets.append(yb.numpy())

all_preds = np.vstack(all_preds)
all_targets = np.vstack(all_targets)

mse = mean_squared_error(all_targets, all_preds)
mae = mean_absolute_error(all_targets, all_preds)
rmse = np.sqrt(mse)
r2 = r2_score(all_targets, all_preds)

print("\n--- Evaluation Metrics ---")
print("MSE:", mse)
print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)


# -----------------------------
# Plot training vs validation
# -----------------------------
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()
