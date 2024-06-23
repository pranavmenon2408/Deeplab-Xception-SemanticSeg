import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import mask
from torchvision import transforms

TRAIN_IMG_DIR = '/home/pranav/DeepLabV3_Xception/data/training/image_2'
TRAIN_MASK_DIR = '/home/pranav/DeepLabV3_Xception/data/training/semantic_rgb'

def get_unique_colors(mask_dir):
    unique_colors = set()
    for mask_name in os.listdir(mask_dir):
        mask_path = os.path.join(mask_dir, mask_name)
        mask = Image.open(mask_path).convert("RGB")
        mask = np.array(mask)
        colors = set(tuple(color) for color in mask.reshape(-1, 3))
        unique_colors.update(colors)
    return list(unique_colors)

def create_color_to_class_mapping(unique_colors):
    return {color: idx for idx, color in enumerate(unique_colors)}

def rgb_to_class_pil(image, color_to_class):
    # Convert PIL image to numpy array
    image_np = np.array(image)
    
    # Create an empty numpy array for the class mask
    class_mask_np = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
    
    # Iterate over the color_to_class dictionary
    for cls, rgb in color_to_class.items():
        # Check where the RGB values match the current RGB tuple
        mask_equal = np.all(image_np == np.array(rgb), axis=-1)
        # Assign class index to corresponding locations in class_mask_np
        class_mask_np[mask_equal] = cls
    
    # Convert numpy array back to PIL image
    class_mask_pil = Image.fromarray(class_mask_np)

    return class_mask_pil

class IndianDrivigDataset(Dataset):
    def __init__(self,img_dir, mask_dir, transform_img=None, transform_mask=None) -> None:
        super(IndianDrivigDataset,self).__init__()
        self.img_dir=img_dir
        self.mask_dir=mask_dir
        self.transform_img=transform_img
        self.transform_mask=transform_mask
        self.sub_dir_img=os.listdir(img_dir)
        self.sub_dir_mask=os.listdir(mask_dir)
        self.images=[]
        self.masks=[]

        for sub_dir in self.sub_dir_img:
            sub_img_dir = os.path.join(img_dir, sub_dir)
            sub_mask_dir = os.path.join(mask_dir, sub_dir)
            
            images = os.listdir(sub_img_dir)
            masks = os.listdir(sub_mask_dir)
            
            # Ensure images and masks are sorted to match each other
            images.sort()
            masks.sort()
            
            # Extend lists with full paths
            self.images.extend([os.path.join(sub_img_dir, img) for img in images])
            self.masks.extend([os.path.join(sub_mask_dir, i) for i in masks])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        image = Image.open(img_path).convert("RGB")
        masks = Image.open(mask_path).convert("RGB")

        if self.transform_img:
            image = self.transform_img(image)

        if self.transform_mask:
            color_to_class={idx: mask.palette[idx].tolist() for idx in range(len(mask.palette))}
            masks=rgb_to_class_pil(masks, color_to_class)
            masks=self.transform_mask(masks)
            
        #print(torch.unique(masks))

        return image, masks
        

    

class RoadSegmentDataset(Dataset):
    def __init__(self, img_dir, mask_dir, color_to_class, transform_img=None, transform_mask=None):
        super(RoadSegmentDataset, self).__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.color_to_class = color_to_class
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.images = os.listdir(img_dir)
        self.masks = os.listdir(mask_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        
        if self.transform_img:
            image = self.transform_img(image)
            #print(image)

        if self.transform_mask:
            #print(mask)

            mask = rgb_to_class_pil(mask, self.color_to_class)
            mask=self.transform_mask(mask)
            #mask = torch.from_numpy(mask).long()
            #print(mask)
        

        return image, mask
    
def get_dataloader(img_dir, mask_dir, batch_size, color_to_class, transform_img=None, transform_mask=None, num_workers=4, pin_memory=True):
    dataset = RoadSegmentDataset(img_dir, mask_dir, color_to_class, transform_img=transform_img, transform_mask=transform_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    return dataloader

def get_dataloader_IDD(img_dir, mask_dir, batch_size, shuffle=True, transform_img=None, transform_mask=None, num_workers=4, pin_memory=True):
    dataset = IndianDrivigDataset(img_dir, mask_dir, transform_img=transform_img, transform_mask=transform_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return dataloader