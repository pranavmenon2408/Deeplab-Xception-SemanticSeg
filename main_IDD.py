import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
from torchmetrics import JaccardIndex
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm
from models.model import DeepLabV3
from models.ccso_net import CCSONet
from models.model_light_2 import LiteDeepLabV3
from utils.utils_IDD import train_loader, DEVICE, NUM_CLASSES, NUM_EPOCHS, LR, calculate_accuracy, val_loader
import os
from itertools import filterfalse as ifilterfalse
from math import isnan


import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_confusion_matrix(preds, targets, num_classes):
    mask = (targets >= 0) & (targets < num_classes)
    return np.bincount(
        num_classes * targets[mask].astype(int) + preds[mask].astype(int),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)

def compute_iou_from_confusion(conf_matrix):
    true_positives = np.diag(conf_matrix)
    false_positives = conf_matrix.sum(axis=0) - true_positives
    false_negatives = conf_matrix.sum(axis=1) - true_positives
    denom = true_positives + false_positives + false_negatives
    iou_per_class = np.divide(
        true_positives, denom, out=np.zeros_like(true_positives, dtype=float), where=denom != 0
    )
    mean_iou = np.mean(iou_per_class)
    return iou_per_class, mean_iou


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
    probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
    labels: [B, H, W] Tensor, ground truth labels (between 0 and C-1)
    classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    per_image: compute the loss per image instead of per batch
    ignore: void class labels
    """
    if per_image:
        loss = mean([lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                     for prob, lab in zip(probas, labels)])
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss

def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
    probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
    labels: [P] Tensor, ground truth labels (between 0 and C-1)
    classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return mean(losses)

def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(LovaszSoftmaxLoss, self).__init__()
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        probas = F.softmax(inputs, dim=1)
        return lovasz_softmax(probas, targets, ignore=self.ignore_index)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        
        # IDD dataset class weights (26 classes: 0-25)
        # These weights are based on typical IDD class distribution and difficulty
        if alpha is None:
            self.alpha = torch.tensor([
                0.8,   # 0: road - typically well-segmented
                1.2,   # 1: sidewalk - moderate difficulty
                0.9,   # 2: building - usually large and well-defined
                4.0,   # 3: wall - can be challenging
                3.5,   # 4: fence - often thin and difficult
                2.5,   # 5: pole - thin objects, moderate difficulty
                5.0,   # 6: traffic light - small objects, higher weight
                2.0,   # 7: traffic sign - moderate size, decent performance
                0.7,   # 8: vegetation - typically large regions, easy
                2.0,   # 9: terrain - moderate difficulty
                0.6,   # 10: sky - usually easy to segment
                1.8,   # 11: person - moderate difficulty
                8.0,   # 12: rider - often small, challenging
                0.9,   # 13: car - common, usually well-segmented
                6.0,   # 14: truck - less common, higher weight
                5.0,   # 15: bus - less frequent, moderate difficulty
                7.0,   # 16: train - rare class, higher weight
                9.0,   # 17: motorcycle - small, challenging
                3.0,   # 18: bicycle - moderate difficulty
                4.0,   # 19: autorickshaw - India-specific, moderate weight
                3.5,   # 20: animal - can be challenging
                2.5,   # 21: curb - thin structure
                1.5,   # 22: parking - moderate difficulty
                6.0,   # 23: railtrack - thin, challenging
                4.5,   # 24: guard rail - barrier-like structure
                2.0    # 25: billboard - moderate difficulty
            ], dtype=torch.float32)
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
            
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C, H, W] - raw logits from model (C should be 26 for IDD)
            targets: [B, H, W] - ground truth labels (0-25 for valid classes)
        """
        # Move alpha to same device as inputs
        if self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)
        
        # Get number of classes (should be 26 for IDD)
        num_classes = inputs.size(1)
        
        # Create valid mask first (before clamping)
        valid_mask = (targets != self.ignore_index).float()
        
        # Clamp targets to valid class range [0, num_classes-1] for valid pixels only
        targets_clamped = targets.clone()
        valid_pixels = (targets != self.ignore_index)
        targets_clamped[valid_pixels] = torch.clamp(targets[valid_pixels], 0, num_classes - 1)
        
        # Compute cross entropy loss using clamped targets
        ce_loss = F.cross_entropy(inputs, targets_clamped, ignore_index=self.ignore_index, reduction='none')
        
        # Compute p_t (probability of true class)
        pt = torch.exp(-ce_loss)
        
        # Initialize alpha weights tensor
        alpha_weights = torch.ones_like(targets_clamped, dtype=torch.float32, device=inputs.device)
        
        # Apply class-specific alpha weights for valid pixels
        for class_idx in range(num_classes):
            if class_idx < len(self.alpha):  # Safety check
                class_mask = (targets_clamped == class_idx) & valid_pixels
                alpha_weights[class_mask] = self.alpha[class_idx]
        
        # Apply focal loss formula: alpha * (1 - p_t)^gamma * CE_loss
        focal_loss = alpha_weights * ((1 - pt) ** self.gamma) * ce_loss * valid_mask
        
        # Return mean loss over valid pixels
        valid_pixel_count = valid_mask.sum().clamp(min=1)
        return focal_loss.sum() / valid_pixel_count



class CombinedLoss(nn.Module):
    def __init__(self, num_classes, ce_weight=0.5, focal_weight=0.3, lovasz_weight=0.2, ignore_index=255):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.focal_loss = FocalLoss(gamma=2.0, ignore_index=ignore_index)
        self.lovasz_loss = LovaszSoftmaxLoss(ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.lovasz_weight = lovasz_weight
    
    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        focal_loss = self.focal_loss(inputs, targets)
        lovasz_loss = self.lovasz_loss(inputs, targets)
        
        total_loss = (self.ce_weight * ce_loss + 
                        self.focal_weight * focal_loss +
                     self.lovasz_weight * lovasz_loss)
        
        return total_loss, ce_loss, lovasz_loss, focal_loss




# Initialize model
model = LiteDeepLabV3(num_classes=NUM_CLASSES, output_stride=16, use_mobilenet_v3=True).to(DEVICE)
#model = DeepLabV3(num_classes=NUM_CLASSES, output_stride=16).to(DEVICE)
criterion = CombinedLoss(num_classes=NUM_CLASSES, ce_weight=1, lovasz_weight=0.5, focal_weight=1, ignore_index=255)
optimizer = optim.Adam(model.parameters(), lr=LR)

train_losses = []
val_losses = []

# Initialize metrics
iou = JaccardIndex(task='multiclass', num_classes=NUM_CLASSES, average="weighted").to(DEVICE)
miou_metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES, ignore_index=255, average="macro", zero_division=1).to(DEVICE)


# Model checkpoint path
model_path = 'deeplabv3_IDD_lite2_iddpart1.pth'
checkpoint_path = 'deeplabv3_IDD_lite2_checkpoint_iddpart1.pth'

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
        total_loss, ce_loss, iou_loss, focal_loss = criterion(output, target)
        total_loss.backward()
        optimizer.step()
        
        loop.set_postfix(total_loss=total_loss.item(), 
        ce_loss=ce_loss.item(), 
        iou_loss=iou_loss.item(),
        focal_loss=focal_loss.item())
        train_loss += total_loss.item()

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    total_iou = 0.0
    total_miou = 0.0
    total_batches = 0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    # For computing per-class IoU via confusion matrix
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    
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
            total_loss, ce_loss, iou_loss, focal_loss = criterion(outputs, masks.squeeze(1).long())
            val_loss += total_loss.item()
            val_accuracy += calculate_accuracy(outputs, masks.squeeze(1))

            iou_val = iou(torch.argmax(outputs, dim=1), masks.squeeze(1).long())
            miou_val = miou_metric(torch.argmax(outputs, dim=1), masks.squeeze(1).long())
            total_iou += iou_val.item()
            total_miou += miou_val.item()
            total_batches += 1

            # Calculate metrics
            predicted_labels = torch.argmax(outputs, dim=1).cpu().numpy()
            true_labels = masks.squeeze(1).cpu().numpy()
            # Update confusion matrix
            flat_preds = predicted_labels.flatten()
            flat_labels = true_labels.flatten()
            confusion_matrix += compute_confusion_matrix(flat_preds, flat_labels, NUM_CLASSES)

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
    total_miou /= total_batches
    total_precision /= total_batches
    total_recall /= total_batches
    total_f1 /= total_batches

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Compute per-class IoU from confusion matrix
    per_class_iou, macro_iou_confmat = compute_iou_from_confusion(confusion_matrix)

    print("\nClass-wise IoU (from confusion matrix):")
    for i in range(NUM_CLASSES):
        print(f"Class {i:2}: IoU = {per_class_iou[i]:.4f}")

    print(f"\nMacro mIoU (confusion matrix): {macro_iou_confmat:.4f}")


    # Print class-wise metrics
    print("\nClass-wise Precision/Recall/F1:")
    for i in range(NUM_CLASSES):
        print(f"Class {i:2}: Precision: {class_precision_avg[i]:.4f}, Recall: {class_recall_avg[i]:.4f}, F1: {class_f1_avg[i]:.4f}")
    
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Mean IOU: {total_iou:.4f}, Mean IOU(macro): {total_miou: .4f}, Precision: {total_precision:.4f}, Recall: {total_recall:.4f}, F1 Score: {total_f1:.4f}')
    
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
print(f"Train losses: {train_losses}")
print(f"Validation losses: {val_losses}")
