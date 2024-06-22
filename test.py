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
model = DeepLabV3(num_classes=35).to(DEVICE)
model.load_state_dict(torch.load('/home/pranav/deeplabv3_IDD.pth'))
model.eval()

# Transformations for the test image
transform = transforms.Compose([
    transforms.Resize((540, 960)),
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
    [160, 82, 45]       # Class 34
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
    # test_img_dir = '/home/pranav/DeepLabV3_Xception/data/testing/image_2'
    # test_mask_out_dir = '/home/pranav/DeepLabV3_Xception/data/testing/mask_result'
    # for img in os.listdir(test_img_dir):
    #     test_image_path = os.path.join(test_img_dir, img)
    #     output_image_path = os.path.join(test_mask_out_dir, img)

    #     predict(test_image_path, output_image_path)
    test_image_path="/home/pranav/DeepLabV3_Xception/idd-20k-II/idd20kII/leftImg8bit/test/200/frame0199_leftImg8bit.jpg"
    output_image_path="/home/pranav/DeepLabV3_Xception/idd-20k-II/idd20kII/mask/test/200/frame0199_mask.png"
    predict(test_image_path, output_image_path)

   
