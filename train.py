import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader 
from torchvision.transforms import ToTensor
from dataset import ConnectomicsDataset
from model import ResidualSymmetricUNet3D

# ハイパーパラメータの設定
batch_size = 4
learning_rate = 0.001
num_epochs = 2

# データセットの読み込み
train_dataset = ConnectomicsDataset('data/', "fold.csv", phase="train")
val_dataset = ConnectomicsDataset("data/", "fold.csv", phase="val")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

# モデル、損失関数、オプティマイザの定義
model = ResidualSymmetricUNet3D(1, 3).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create a GradScaler for mixed precision training
scaler = torch.cuda.amp.GradScaler()

# トレーニングループ
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        
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
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')
    
    # 検証
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.cuda(), labels.cuda()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    print(f'Validation Loss: {val_loss/len(val_loader)}')

torch.save(model.state_dict(), 'residual_symmetric_unet3d.pth')
