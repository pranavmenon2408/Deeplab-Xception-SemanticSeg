import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
from torchmetrics import JaccardIndex
from tqdm import tqdm
from models.model import DeepLabV3
from models.ccso_net import CCSONet
from models.model_light import LiteDeepLabV3
from utils.utils_IDD import train_loader, DEVICE, NUM_CLASSES, NUM_EPOCHS, LR, calculate_accuracy, val_loader
import os

# Initialize model
model = LiteDeepLabV3(num_classes=NUM_CLASSES, output_stride=16, use_hierarchical_aspp=True).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Initialize metrics
iou = JaccardIndex(task='multiclass', num_classes=NUM_CLASSES, average="weighted").to(DEVICE)

# Model checkpoint path
model_path = 'deeplabv3_IDD_best_CCAR_and_ACDSC_lite_iddpart1.pth'
checkpoint_path = 'deeplabv3_IDD_checkpoint_lite_iddpart1.pth'

# Initialize variables
best_iou = 0.0
start_epoch = 0

# Load existing model if it exists
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load training progress
    start_epoch = checkpoint['epoch'] + 1
    best_iou = checkpoint['best_iou']
    
    print(f"Resuming training from epoch {start_epoch} with best IoU: {best_iou:.4f}")
    
elif os.path.exists(model_path):
    print(f"Loading best model weights from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print("Starting fresh training with pre-trained weights")
else:
    print("Starting training from scratch")

# Initialize class-wise metrics
class_precision_sum = np.zeros(NUM_CLASSES)
class_recall_sum = np.zeros(NUM_CLASSES)
class_f1_sum = np.zeros(NUM_CLASSES)
class_counts = np.zeros(NUM_CLASSES)

# Training loop
for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
    
    for idx, (data, target) in enumerate(loop):
        data, target = data.to(DEVICE), target.to(DEVICE)
        target = target.squeeze(1)
        target = target.long()
        
        # Handle invalid target values
        count_invalid_train = (target > 26).sum().item()
        if count_invalid_train > 0:
            print(f"Epoch {epoch + 1} | Batch {idx + 1}: Found {count_invalid_train} invalid values (>26) in training targets.")
            target[target > 26] = 26
        
        assert target.max().item() < NUM_CLASSES, f"Target contains an invalid class index, {target.max().item()}"
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        loop.set_postfix(loss=loss.item())
        train_loss += loss.item()

    # Validation phase
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

            # Handle invalid mask values
            count_invalid_val = (masks > 26).sum().item()
            if count_invalid_val > 0:
                print(f"Epoch {epoch + 1} | Validation Batch {batch_idx + 1}: Found {count_invalid_val} invalid values (>26) in validation masks. Replacing them with 26.")
                masks[masks > 26] = 26
            
            outputs = model(images)
            loss = criterion(outputs, masks.squeeze(1).long())
            
            val_loss += loss.item()
            val_accuracy += calculate_accuracy(outputs, masks.squeeze(1))

            iou_val = iou(torch.argmax(outputs, dim=1), masks.squeeze(1).long())
            total_iou += iou_val.item()
            total_batches += 1

            # Calculate metrics
            predicted_labels = torch.argmax(outputs, dim=1).cpu().numpy()
            true_labels = masks.squeeze(1).cpu().numpy()
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels.flatten(), predicted_labels.flatten(), 
                average='weighted', zero_division=1
            )
            
            total_precision += precision
            total_recall += recall
            total_f1 += f1

            # Class-wise metrics
            precision_c, recall_c, f1_c, support_c = precision_recall_fscore_support(
                true_labels.flatten(), predicted_labels.flatten(),
                labels=np.arange(NUM_CLASSES), average=None, zero_division=0
            )

            # Accumulate class-wise metrics
            class_precision_sum += precision_c
            class_recall_sum += recall_c
            class_f1_sum += f1_c
            class_counts += (support_c > 0)

    # Calculate averages
    class_precision_avg = np.divide(class_precision_sum, class_counts, where=class_counts!=0)
    class_recall_avg = np.divide(class_recall_sum, class_counts, where=class_counts!=0)
    class_f1_avg = np.divide(class_f1_sum, class_counts, where=class_counts!=0)

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    val_accuracy /= len(val_loader)
    total_iou /= total_batches
    total_precision /= total_batches
    total_recall /= total_batches
    total_f1 /= total_batches

    # Print class-wise metrics
    print("\nClass-wise Precision/Recall/F1:")
    for i in range(NUM_CLASSES):
        print(f"Class {i:2}: Precision: {class_precision_avg[i]:.4f}, Recall: {class_recall_avg[i]:.4f}, F1: {class_f1_avg[i]:.4f}")
    
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Mean IOU: {total_iou:.4f}, Precision: {total_precision:.4f}, Recall: {total_recall:.4f}, F1 Score: {total_f1:.4f}')
    
    # Save best model
    if total_iou > best_iou:
        best_iou = total_iou
        torch.save(model.state_dict(), model_path)
        print('Best model saved with Mean IOU: ', best_iou)
    
    # Save checkpoint every epoch
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_iou': best_iou,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'total_iou': total_iou
    }
    torch.save(checkpoint, checkpoint_path)
    
    # Reset class-wise metrics for next epoch
    class_precision_sum = np.zeros(NUM_CLASSES)
    class_recall_sum = np.zeros(NUM_CLASSES)
    class_f1_sum = np.zeros(NUM_CLASSES)
    class_counts = np.zeros(NUM_CLASSES)

print(f"Training completed! Best IoU achieved: {best_iou:.4f}")
