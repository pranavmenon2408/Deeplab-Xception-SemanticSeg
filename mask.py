import os
import cv2
import numpy as np
import skimage
import json

INPUT_TRAIN_IMG_DIR="/home/pranav/DeepLabV3_Xception/idd-20k-II/idd20kII/leftImg8bit/train"
INPUT_TRAIN_JSON_DIR="/home/pranav/DeepLabV3_Xception/idd-20k-II/idd20kII/gtFine/train"
OUTPUT_TRAIN_MASK_DIR="/home/pranav/DeepLabV3_Xception/idd-20k-II/idd20kII/mask/train"

INPUT_VAL_JSON_DIR="/home/pranav/DeepLabV3_Xception/idd-20k-II/idd20kII/gtFine/val"
OUTPUT_VAL_MASK_DIR="/home/pranav/DeepLabV3_Xception/idd-20k-II/idd20kII/mask/val"

num_classes = 34  # Number of classes in your dataset

labels={}

palette = np.array([
    [0, 70, 0],          # Class 0 (Background)
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
    [160, 82, 45],
    [64, 135, 208],
    [60, 105, 85],
    [200, 56, 105],
    [100, 150, 140],
    [12, 200, 140],
    [90, 200, 50],
    [35, 145, 235]# Class 34
], dtype=np.uint8)


def create_mask(img_dir, json_dir, mask_dir):
    class_id = 1
    for sub_dir_1 in os.listdir(json_dir):
        for json_file in os.listdir(os.path.join(json_dir, sub_dir_1)):
            with open(os.path.join(json_dir, sub_dir_1, json_file)) as f:
                data = json.load(f)
                img = np.zeros((1080,1920), dtype=np.uint8)
                for obj in data['objects']:
                    if obj['label'] not in labels:
                        labels[obj['label']] = class_id
                        class_id += 1
                    
                    if 'polygon' in obj:
                        pts = np.array(obj['polygon'], np.int32)
                        if pts.shape[0] > 0:
                            center_x = np.mean(pts[:, 0])
                            center_y = np.mean(pts[:, 1])
                            flipped_rotated_pts = np.fliplr(pts - np.array([center_x, center_y]))[::-1, :]
                            flipped_rotated_pts += np.array([center_y, center_x])

                            # Create mask with flipped and rotated points
                            mask = skimage.draw.polygon2mask(image_shape=(1080, 1920), polygon=flipped_rotated_pts)

                            # Assign label to masked pixels
                            img[mask] = labels[obj['label']]
                    
                print(img)
                print(np.unique(img))
                print(f"Saving mask {os.path.join(mask_dir, sub_dir_1, json_file.replace('.json', '.jpg'))}")
                os.makedirs(os.path.dirname(os.path.join(mask_dir, sub_dir_1, json_file.replace('.json', '.jpg'))), exist_ok=True)
                cv2.imwrite(os.path.join(mask_dir, sub_dir_1, json_file.replace('.json', '.jpg')), img)

create_mask(INPUT_TRAIN_IMG_DIR, INPUT_TRAIN_JSON_DIR, OUTPUT_TRAIN_MASK_DIR)