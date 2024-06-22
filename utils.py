import os
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import functional as F
import dataset
from dataset import get_dataloader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from dataset import TRAIN_MASK_DIR, TRAIN_IMG_DIR, create_color_to_class_mapping, get_unique_colors

def calculate_accuracy(preds, labels):
    preds = torch.argmax(preds, dim=1)  # Get predicted class indices
    correct = torch.sum(preds == labels)
    total = labels.numel()
    accuracy = correct.item() / total
    return accuracy



LR=1e-4
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE=8
NUM_EPOCHS=30
NUM_CLASSES=29
NUM_WORKERS=4
PIN_MEMORY=True

train_img_transforms =  transforms.Compose(
    [
        transforms.Resize((312, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


train_mask_transforms =  transforms.Compose(
    [
        transforms.Resize((312, 1024), interpolation=Image.NEAREST),
        transforms.PILToTensor(),
    ]
)

#unique_colors = get_unique_colors(TRAIN_MASK_DIR)
color_to_class = {(102, 102, 156): 0, (107, 142, 35): 1, (0, 0, 142): 2, (220, 220, 0): 3, (70, 130, 180): 4, (0, 0, 90): 5, (255, 0, 0): 6, (150, 100, 100): 7, (0, 0, 230): 8, (220, 20, 60): 9, (244, 35, 232): 10, (128, 64, 128): 11, (111, 74, 0): 12, (150, 120, 90): 13, (70, 70, 70): 14, (250, 170, 30): 15, (119, 11, 32): 16, (0, 80, 100): 17, (152, 251, 152): 18, (0, 0, 110): 19, (0, 60, 100): 20, (0, 0, 0): 21, (190, 153, 153): 22, (0, 0, 70): 23, (81, 0, 81): 24, (153, 153, 153): 25, (230, 150, 140): 26, (250, 170, 160): 27, (180, 165, 180): 28}



train_loader = get_dataloader(dataset.TRAIN_IMG_DIR, dataset.TRAIN_MASK_DIR, BATCH_SIZE,color_to_class, transform_img=train_img_transforms,transform_mask=train_mask_transforms, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)


dataset_size = len(train_loader.dataset)
val_size = int(0.2 * dataset_size)  # 10% of the dataset for validation

indices = list(range(dataset_size))
np.random.shuffle(indices)

train_indices, val_indices = indices[val_size:], indices[:val_size]

train_dataset = Subset(train_loader.dataset, train_indices)
val_dataset = Subset(train_loader.dataset, val_indices)

final_train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
final_val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
