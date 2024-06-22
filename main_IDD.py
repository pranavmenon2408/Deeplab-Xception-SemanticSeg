import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from model import DeepLabV3
from utils_IDD import train_loader, DEVICE, NUM_CLASSES, NUM_EPOCHS, LR, calculate_accuracy,val_loader

model = DeepLabV3(num_classes=35).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    loop = tqdm(train_loader)
    for idx, (data, target) in enumerate(loop):
        data, target = data.to(DEVICE), target.to(DEVICE)
        target=target.squeeze(1)
        target=target.long()
        #print(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

       
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            #print(masks)
            outputs = model(images)
            #print(outputs)
            loss = criterion(outputs, masks.squeeze(1).long())
            
            val_loss += loss.item()
            val_accuracy += calculate_accuracy(outputs, masks.squeeze(1))

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    val_accuracy /= len(val_loader)
    
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
torch.save(model.state_dict(), 'deeplabv3_IDD.pth')

