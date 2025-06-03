import os
import cv2
import numpy as np
import skimage
import json

INPUT_TRAIN_IMG_DIR="/data/pranav/idd20kII/leftImg8bit/train"
INPUT_TRAIN_JSON_DIR="/data/pranav/idd20kII/gtFine/train"
OUTPUT_TRAIN_MASK_DIR ="/data/pranav/idd20kII/mask/train"

INPUT_VAL_JSON_DIR="/data/pranav/idd20kII/gtFine/val"
OUTPUT_VAL_MASK_DIR="/data/pranav/idd20kII/mask/val"


labels={'road': 0, 'sky': 25, 'drivable fallback': 1, 'vehicle fallback': 12, 'non-drivable fallback': 3, 'curb': 13, 'obs-str-bar-fallback': 21, 'vegetation': 24, 'pole': 20, 'billboard': 17, 'building': 22, 'truck': 10, 'wall': 14, 'rider': 5, 'motorcycle': 6, 'autorickshaw': 8, 'car': 9, 'person': 4, 'fence': 15, 'traffic sign': 18, 'rectification border': 2, 'bicycle': 7, 'bus': 11, 'fallback background': 25, 'polegroup': 20, 'sidewalk': 2, 'bridge': 23, 'animal': 4, 'traffic light': 19, 'out of roi': 26, 'caravan': 12, 'guard rail': 16, 'rail track': 3, 'trailer': 12, 'parking': 1, 'unlabeled': 26, 'tunnel': 23, 'train': 26, 'ego vehicle': 9, 'ground': 2, 'license plate': 26}



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