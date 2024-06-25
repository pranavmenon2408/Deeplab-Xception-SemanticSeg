import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
from torchmetrics import JaccardIndex
from tqdm import tqdm
from model import DeepLabV3
from utils_IDD import train_loader, DEVICE, NUM_CLASSES, NUM_EPOCHS, LR, calculate_accuracy, val_loader

model = DeepLabV3(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

iou=JaccardIndex(task='multiclass',num_classes=NUM_CLASSES, average="weighted").to(DEVICE)



for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    loop = tqdm(train_loader)
    for idx, (data, target) in enumerate(loop):
        data, target = data.to(DEVICE), target.to(DEVICE)
        target=target.squeeze(1)
        target=target.long()
        assert target.max().item() < NUM_CLASSES, f"Target contains an invalid class index, {target.max().item()}"
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
    total_iou = 0.0
    total_batches = 0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
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

            iou_val=iou(torch.argmax(outputs, dim=1), masks.squeeze(1).long())
            total_iou += iou_val.item()
            total_batches += 1

            predicted_labels = torch.argmax(outputs, dim=1).cpu().numpy()
            true_labels = masks.squeeze(1).cpu().numpy()
            precision, recall, f1, _ = precision_recall_fscore_support(true_labels.flatten(), predicted_labels.flatten(), average='weighted', zero_division=1)
            
            total_precision += precision
            total_recall += recall
            total_f1 += f1


    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    val_accuracy /= len(val_loader)
    total_iou /= total_batches
    total_precision /= total_batches
    total_recall /= total_batches
    total_f1 /= total_batches
    
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Mean IOU: {total_iou:.4f}, Precision: {total_precision:.4f}, Recall: {total_recall:.4f}, F1 Score: {total_f1:.4f}')

torch.save(model.state_dict(), 'deeplabv3_IDD.pth')

