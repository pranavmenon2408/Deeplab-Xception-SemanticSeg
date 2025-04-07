import os
import torch
import torch.nn.functional as F
from torchvision import transforms
import time
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from models.model import DeepLabV3
from utils.utils_IDD import DEVICE, NUM_CLASSES

start_event=torch.cuda.Event(enable_timing=True)
end_event=torch.cuda.Event(enable_timing=True)

model = DeepLabV3(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load('/home/pranav/DeepLabV3_Xception/deeplabv3_IDD_best_again_2.pth', weights_only=True))
model.eval()

# Updated transform pipeline
transform = transforms.Compose([
    transforms.Resize((720, 1280)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



# Function to visualize predictions
def visualize_prediction(image, mask, save_path=None):
    image = image.permute(1, 2, 0).cpu().numpy()
    mask = mask.cpu().numpy().astype(np.uint8)
    palette = np.array([
    [0, 0, 0],          # Class 0 (Background)
    [102, 102, 156],    # Class 1
    [107, 142, 35],     # Class 2
    [0, 0, 142],        # Class 3
    [220, 220, 0],      # Class 4
    [70, 130, 180],     # Class 5
    [0, 0, 90],         # Class 6
    [255, 0, 0],        # Class 7
    [150, 100, 100],    # Class 8
    [0, 0, 230],        # Class 9
    [220, 20, 60],      # Class 10
    [244, 35, 232],     # Class 11
    [128, 64, 128],     # Class 12
    [111, 74, 0],       # Class 13
    [70, 70, 70],       # Class 14
    [250, 170, 30],     # Class 15
    [119, 11, 32],      # Class 16
    [0, 80, 100],       # Class 17
    [152, 251, 152],    # Class 18
    [0, 0, 110],        # Class 19
    [0, 60, 100],       # Class 20
    [0, 0, 70],         # Class 21
    [190, 153, 153],    # Class 22
    [81, 0, 81],        # Class 23
    [153, 153, 153],    # Class 24
    [230, 150, 140],    # Class 25
    [250, 170, 160],    # Class 26
    [180, 165, 180],    # Class 27
    [220, 220, 220],    # Class 28
    [0, 250, 154],      # Class 29
    [105, 105, 105],    # Class 30
    [0, 255, 255],      # Class 31
    [64, 224, 208],     # Class 32
    [255, 20, 147],     # Class 33
    [160, 82, 45],      # Class 34
    [255, 165, 0],      # Class 35 (New)
    [75, 0, 130],       # Class 36 (New)
    [255, 255, 0],      # Class 37 (New)
    [0, 128, 128],      # Class 38 (New)
    [255, 105, 180],    # Class 39 (New)
    [123, 104, 238],    # Class 40 (New)
    [255, 69, 0],       # Class 41 (New)
    [154, 205, 50],     # Class 42 (New)
    [72, 209, 204]      # Class 43 (New)
], dtype=np.uint8)

    mask_colored = palette[mask]
    if save_path:
        mask_colored_image = Image.fromarray(mask_colored)
        mask_colored_image.save(save_path)

# Function for batch-wise prediction
def predict_batch(image_paths, output_paths):
    batch_size = 1  # Adjust batch size as per your system's capability

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_images = []
            batch_input_paths = []
            for j in range(batch_size):
                idx = i + j
                if idx < len(image_paths):
                    image_path = image_paths[idx]
                    image = Image.open(image_path).convert('RGB')
                    input_image = transform(image)
                    batch_images.append(input_image)
                    batch_input_paths.append(image_path)

            if len(batch_images) > 0:
                batch_images = torch.stack(batch_images).to(DEVICE)
                start_event.record()
                output = model(batch_images)
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event)/1000
                print(f"Inference time for batch {i // batch_size}: {elapsed_time} seconds")

                for j, image_path in enumerate(batch_input_paths):
                    output_j = output[j].unsqueeze(0)
                    output_j = F.interpolate(output_j, size= image.size[::-1], mode='bilinear', align_corners=True)
                    output_j = torch.argmax(output_j, dim=1).squeeze(0)
                    output_image_path = output_paths[i + j]
                    visualize_prediction(batch_images[j], output_j, output_image_path)

# Main function
if __name__ == '__main__':
    test_img_dir = '/data/pranav/IDD_Segmentation/leftImg8bit/val'
    test_mask_out_dir = '/data/pranav/IDD_Segmentation/mask/test'
    
    image_paths = []
    output_paths = []

    # Collect all image paths and corresponding output paths
    for sub_dir in os.listdir(test_img_dir):
        for img in os.listdir(os.path.join(test_img_dir, sub_dir)):
            image_paths.append(os.path.join(test_img_dir, sub_dir, img))
            output_paths.append(os.path.join(test_mask_out_dir, sub_dir, img))
            os.makedirs(os.path.dirname(output_paths[-1]), exist_ok=True)

    # Perform batch-wise prediction
    predict_batch(image_paths, output_paths)