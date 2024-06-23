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

def create_mask(json_dir, mask_dir):
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

create_mask(INPUT_TRAIN_JSON_DIR, OUTPUT_TRAIN_MASK_DIR)