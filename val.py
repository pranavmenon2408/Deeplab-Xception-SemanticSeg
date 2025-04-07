import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from torchmetrics import JaccardIndex
from utils.utils_IDD import train_loader, DEVICE, NUM_CLASSES, NUM_EPOCHS, LR, calculate_accuracy, val_loader
from models.model import DeepLabV3
import torch.nn as nn
import torch.optim as optim

iou=JaccardIndex(task='multiclass',num_classes=NUM_CLASSES, average="weighted").to(DEVICE)

model = DeepLabV3(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load("/home/pranav/deeplabv3_IDD.pth", weights_only=True))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_iou = 0.0
val_loss = 0.0
val_accuracy = 0.0
total_iou = 0.0
total_batches = 0
total_precision = 0.0
total_recall = 0.0
total_f1 = 0.0


with torch.no_grad():
    print(len(val_loader))
    for batch_idx, (images, masks) in enumerate(val_loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        count_invalid_val = (masks > 41).sum().item()
        if count_invalid_val > 0:
            #print(f"Epoch {epoch + 1} | Validation Batch {batch_idx + 1}: Found {count_invalid_val} invalid values (>41) in validation masks. Replacing them with 36.")
            masks[masks > 41] = 36
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
    val_loss /= len(val_loader)
    val_accuracy /= len(val_loader)
    total_iou /= total_batches
    total_precision /= total_batches
    total_recall /= total_batches
    total_f1 /= total_batches
    print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f} | Validation IoU: {total_iou:.4f} | Validation Precision: {total_precision:.4f} | Validation Recall: {total_recall:.4f} | Validation F1: {total_f1:.4f}")