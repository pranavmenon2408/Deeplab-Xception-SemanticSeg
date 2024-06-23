import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import functional as F
from dataset import get_dataloader_IDD
import mask
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

def calculate_accuracy(preds, labels):
    preds = torch.argmax(preds, dim=1)  # Get predicted class indices
    correct = torch.sum(preds == labels)
    total = labels.numel()
    accuracy = correct.item() / total
    return accuracy


INPUT_TRAIN_IMAGE_DIR="/home/pranav/DeepLabV3_Xception/idd-20k-II/idd20kII/leftImg8bit/train"
INPUT_TRAIN_MASK_DIR="/home/pranav/DeepLabV3_Xception/idd-20k-II/idd20kII/mask/train"

INPUT_VAL_IMAGE_DIR="/home/pranav/DeepLabV3_Xception/idd-20k-II/idd20kII/leftImg8bit/val"
INPUT_VAL_MASK_DIR="/home/pranav/DeepLabV3_Xception/idd-20k-II/idd20kII/mask/val"


LR=1e-4
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE=16
NUM_EPOCHS=5
NUM_CLASSES=len(mask.palette)
NUM_WORKERS=4
PIN_MEMORY=True

train_img_transforms =  transforms.Compose(
    [
        transforms.Resize((320, 540)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


train_mask_transforms =  transforms.Compose(
    [
        transforms.Resize((320, 540)),
        transforms.PILToTensor(),
    ]
)




train_loader = get_dataloader_IDD(INPUT_TRAIN_IMAGE_DIR, INPUT_TRAIN_MASK_DIR, batch_size=BATCH_SIZE, transform_img=train_img_transforms,transform_mask=train_mask_transforms, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
val_loader= get_dataloader_IDD(INPUT_VAL_IMAGE_DIR, INPUT_VAL_MASK_DIR, batch_size=1, shuffle=False, transform_img=train_img_transforms,transform_mask=train_mask_transforms, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

