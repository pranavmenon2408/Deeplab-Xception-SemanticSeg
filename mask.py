import os
import cv2
import numpy as np
import skimage
import json

INPUT_TRAIN_IMG_DIR="/data/pranav/IDD_Segmentation/leftImg8bit/train"
INPUT_TRAIN_JSON_DIR="/data/pranav/IDD_Segmentation/gtFine/train"
OUTPUT_TRAIN_MASK_DIR="/data/pranav/IDD_Segmentation/mask/train"

INPUT_VAL_JSON_DIR="/data/pranav/IDD_Segmentation/gtFine/val"
OUTPUT_VAL_MASK_DIR="/data/pranav/IDD_Segmentation/mask/val"


labels={'road': 1, 'sky': 2, 'drivable fallback': 3, 'vehicle fallback': 4, 'non-drivable fallback': 5, 'curb': 6, 'obs-str-bar-fallback': 7, 'vegetation': 8, 'pole': 9, 'billboard': 10, 'building': 11, 'truck': 12, 'wall': 13, 'rider': 14, 'motorcycle': 15, 'autorickshaw': 16, 'car': 17, 'person': 18, 'fence': 19, 'traffic sign': 20, 'rectification border': 21, 'bicycle': 22, 'bus': 23, 'fallback background': 24, 'polegroup': 25, 'sidewalk': 26, 'bridge': 27, 'animal': 28, 'traffic light': 29, 'out of roi': 30, 'caravan': 31, 'guard rail': 32, 'rail track': 33, 'trailer': 34, 'parking': 35, 'unlabeled': 36, 'tunnel': 37, 'train': 38, 'ego vehicle': 39, 'ground': 40, 'license plate': 41}



def create_mask(json_dir, mask_dir):
    class_id = len(labels)
    for sub_dir_1 in os.listdir(json_dir):
        for json_file in os.listdir(os.path.join(json_dir, sub_dir_1)):
            with open(os.path.join(json_dir, sub_dir_1, json_file)) as f:
                data = json.load(f)
                h = data['imgHeight']
                w = data['imgWidth']
                img = np.zeros((h, w), dtype=np.uint8)
                for obj in data['objects']:
                    
                    if 'polygon' in obj:
                        pts = np.array(obj['polygon'], np.int32)
                        if pts.shape[0] > 0:
                            center_x = np.mean(pts[:, 0])
                            center_y = np.mean(pts[:, 1])
                            flipped_rotated_pts = np.fliplr(pts - np.array([center_x, center_y]))[::-1, :]
                            flipped_rotated_pts += np.array([center_y, center_x])

                            # Create mask with flipped and rotated points
                            mask = skimage.draw.polygon2mask(image_shape=(h, w), polygon=flipped_rotated_pts)

                            # Assign label to masked pixels
                            if obj['label'] not in labels:
                                img[mask] = labels['unlabeled']
                            else:
                                img[mask] = labels[obj['label']]
                    
                print(img)
                print(np.unique(img))
                print(f"Saving mask {os.path.join(mask_dir, sub_dir_1, json_file.replace('.json', '.jpg'))}")
                os.makedirs(os.path.dirname(os.path.join(mask_dir, sub_dir_1, json_file.replace('.json', '.jpg'))), exist_ok=True)
                cv2.imwrite(os.path.join(mask_dir, sub_dir_1, json_file.replace('.json', '.jpg')), img)
    print(labels)
#
create_mask(INPUT_TRAIN_JSON_DIR, OUTPUT_TRAIN_MASK_DIR)
create_mask(INPUT_VAL_JSON_DIR, OUTPUT_VAL_MASK_DIR)