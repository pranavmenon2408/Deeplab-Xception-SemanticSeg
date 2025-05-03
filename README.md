# DeepLab-Xception Semantic Segmentation

This repository contains an implementation of the DeepLab v3+ architecture using an Aligned Xception backbone for performing semantic segmentation. It is designed for urban scene understanding and works well with datasets like the Indian Driving Dataset or Cityscapes.

---

## 🔍 Overview

DeepLab v3+ is a state-of-the-art semantic segmentation model that combines atrous spatial pyramid pooling (ASPP) with a decoder module for precise object boundary delineation. This repo uses an Aligned Xception backbone for improved feature extraction.

---

## 🏗️ Architecture

- **Backbone:** Aligned Xception  
- **ASPP Module:** Multi-scale context aggregation  
- **Decoder:** Upsampling with skip connections for fine spatial details  

---

## ⚙️ Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/DeepLab-Xception.git
   cd DeepLab-Xception
   ```
2. Create a Python virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```
Using the Indian Driving Dataset (IDD)
To use the IDD dataset with this implementation:

1. Download and extract the IDD dataset into the dataset/ directory:
   ```bash
   dataset/
    └── IDD_Segmentation/
        ├── leftImg8bit/
        ├── gtFine/
        └── annotations/  # JSON files, if using original annotations
   ```

2. If you need to generate segmentation masks from the .json annotation files, run:

   ```bash
    python mask.py
   ```
⚠️ Before running, edit the dataset path in mask.py to point to your local dataset directory.

3. Update the train and val paths in:

    ```bash
      utils/utils_IDD.py
    ```
4. To train the model on IDD:

    ```bash
      python main_IDD.py
    ```
5. To evaluate the model and generate segmentation maps on the test dataset:

    ```bash
      python test.py
    ```
Output segmented images will be saved in the specified output folder.

## 📊 Results

| Dataset                          | Train Loss | Val Loss | Val Accuracy | Mean IoU | Precision | Recall | F1 Score |
|----------------------------------|------------|----------|---------------|----------|-----------|--------|----------|
| IDD-Segmentation (20K) - Part 1  | 0.2394     | 0.6325   | 0.8570        | 0.7891   | 0.8800    | 0.8570 | 0.8522   |
| IDD-Segmentation (20K) - Part 2  | 0.2029     | 0.6869   | 0.8340        | 0.7324   | 0.8489    | 0.8102 | 0.8051   |

## 🖼️ Sample Results

Left: Input Image | Right: Predicted Segmentation Map

---

### Example 1
![Input](https://github.com/user-attachments/assets/a81beab2-c58e-43b6-88dd-7471a51825f5)
![Output](https://github.com/user-attachments/assets/cd9e67d1-5ec6-4fdb-8119-8a62feb48384)

### Example 2
![Input](https://github.com/user-attachments/assets/d3435f5e-6098-4351-a438-d4887e98ca2f)
![Output](https://github.com/user-attachments/assets/eb6a5495-16b7-4c05-9be9-b7650e0d35b1)


