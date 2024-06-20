import os
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from model import DeepLabV3
from utils import DEVICE, NUM_CLASSES

# Load the trained model
model = DeepLabV3(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load('/home/pranav/deeplabv3_final.pth'))
model.eval()

# Transformations for the test image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def visualize_prediction(image, mask, save_path=None):
    # Convert the image and mask to numpy arrays
    image = image.permute(1, 2, 0).cpu().numpy()
    print(image.shape)
    mask = mask.cpu().numpy().astype(np.uint8)  # Ensure mask is uint8

    # Create a color palette for the mask
    palette = np.array([
    [0, 0, 0],         # Class 0
    [128, 0, 0],       # Class 1
    [0, 128, 0],       # Class 2
    [128, 128, 0],     # Class 3
    [0, 0, 128],       # Class 4
    [128, 0, 128],     # Class 5
    [0, 128, 128],     # Class 6
    [128, 128, 128],   # Class 7
    [64, 0, 0],        # Class 8
    [192, 0, 0],       # Class 9
    [64, 128, 0],      # Class 10
    [192, 128, 0],     # Class 11
    [64, 0, 128],      # Class 12
    [192, 0, 128],     # Class 13
    [64, 128, 128],    # Class 14
    [192, 128, 128],   # Class 15
    [0, 64, 0],        # Class 16
    [128, 64, 0],      # Class 17
    [0, 192, 0],       # Class 18
    [128, 192, 0],     # Class 19
    [0, 64, 128],      # Class 20
    [128, 64, 128],    # Class 21
    [0, 192, 128],     # Class 22
    [128, 192, 128],   # Class 23
    [64, 64, 0],       # Class 24
    [192, 64, 0],      # Class 25
    [64, 192, 0],      # Class 26
    [192, 192, 0],     # Class 27
    [64, 64, 128],     # Class 28
], dtype=np.uint8)

    mask_colored = palette[mask]

    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.title('Input Image')
    # plt.imshow(image)
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.title('Predicted Mask')
    # plt.imshow(mask_colored)
    # plt.axis('off')

    if save_path:
        mask_colored_image = Image.fromarray(mask_colored)
        mask_colored_image.save(save_path)
    
    #plt.show()

def predict(image_path, save_path=None):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    input_image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_image)
        output = F.interpolate(output, size=image.size[::-1], mode='bilinear', align_corners=True)
        output = torch.argmax(output, dim=1).squeeze(0)

        # Debugging step: print unique values in the output mask
        unique_values = torch.unique(output)
        print(f"Unique values in the output mask: {unique_values}")

    visualize_prediction(input_image.squeeze(0), output, save_path)

if __name__ == '__main__':
    test_img_dir = '/home/pranav/DeepLabV3_Xception/data/testing/image_2'
    test_mask_out_dir = '/home/pranav/DeepLabV3_Xception/data/testing/mask_result'
    for img in os.listdir(test_img_dir):
        test_image_path = os.path.join(test_img_dir, img)
        output_image_path = os.path.join(test_mask_out_dir, img)

        predict(test_image_path, output_image_path)
        

   
