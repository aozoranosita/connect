import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader 
from torchvision.transforms import ToTensor
from dataset import ConnectomicsDataset
from model import ResidualSymmetricUNet3D
import pickle

# ハイパーパラメータの設定
batch_size = 16
learning_rate = 0.004
num_epochs = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# データセットの読み込み
train_dataset = ConnectomicsDataset('data/', "fold.csv", phase="train")
val_dataset = ConnectomicsDataset("data/", "fold.csv", phase="val")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

# モデル、損失関数、オプティマイザの定義
model = ResidualSymmetricUNet3D(1, 3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create a GradScaler for mixed precision training
scaler = torch.cuda.amp.GradScaler()

# トレーニングループ
Loss, Val_loss = [], []
min_loss = 1 << 15
best = 0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        
        optimizer.zero_grad()
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        # Backward pass with mixed precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #loss.backward()
        #optimizer.step()
        
        running_loss += loss.item()
    loss = running_loss/len(train_loader)
    Loss.append(loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss}')
    
    # evaluation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            val_loss += loss.item()
    loss = val_loss/len(val_loader)
    if loss < min_loss:
        best = epoch
        min_loss = loss
        best_weight = model.state_dict()

    Val_loss.append(loss)
    print(f'Validation Loss: {loss}')
    
    if epoch - best > 6: #early stopping
        break

torch.save(best_weight, f"model/Unet3d_{best}.pth")


loss_path = 'model/loss.pkl'
with open(loss_path, 'wb') as file:
    pickle.dump((Loss, Val_loss), file)
