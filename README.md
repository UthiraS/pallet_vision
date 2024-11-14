# Pallet Detection and Segmentation
This project implements an end-to-end solution for pallet detection and segmentation using YOLOv11 and SAM2 (Segment Anything Model 2). It includes tools for annotation, training, inference, and evaluation.

![Pallet Detection Example](results/aug_pallet_287.jpg)
![Pallet Segmentation Example](results/vis_pallet_337.png)

## Dataset
The complete dataset is available on Google Drive:
- [Dataset](https://drive.google.com/drive/u/1/folders/1OIxiRhRfp1iDgKj_7L7DgIXOJ6OHqHTa)

## Fine tuned model Weights
Fine tuned model Weights on Google Drive:
- [Weights](https://drive.google.com/drive/u/1/folders/1L9P7ImR7rIuDuT4IzY570NkvcNi-QZVT)


## Installation
1. Create and activate a virtual environment:
```bash
python -m venv venv_sam2
source venv_sam2/bin/activate # Linux/Mac
# or
.\venv_sam2\Scripts\activate # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
## Requirements
```txt
torch>=2.0.0
ultralytics>=8.0.0
segment-anything-2==0.1.0
supervision>=0.3.0
opencv-python>=4.8.0
numpy>=1.24.0
ros2 humble
```


## Project Structure
```
pallet_vision/
├── data/
│   ├── dataset/             # Detection dataset
│   └── dataset-segment/     # Segmentation dataset
├── models/
│   └── weights/             # Trained model weights
├── results/                 # Inference results
├── scripts/
│   ├── detection/
│   │   ├── annotation/
│   │   │   └── pallet_annotator.py
│   │   ├── network/
│   │   │   └── pallet_object_detection.py
│   │   └── inference/
│   │       └── detection_inference_display.py      # ✓ Available
│   ├── segmentation/
│   │   ├── annotation/
│   │   │   ├── sam2_pallets.py
│   │   │   └── det2_segformat.py
│   │   ├── network/
│   │   │   └── segment_network.py
│   │   └── inference/
│   │       └── segmentation_inference_display.py   # ✓ Available
│   └── utils/
│       ├── yolo_label_normalizer.py
│       ├── dataset_preparation.py
│       ├── datasize.py
│       ├── dataset_augment.py
│       ├── dataset_download.py
│       └── optimize_yolo.py
└── requirements.txt
```

## Available Scripts

### Detection Inference
Run single-image detection inference with live visualization:
```bash
 python /home/uthira/pallet-detection/scripts/detection/inference/detection_inference_display.py --weights /home/uthira/pallet-detection/models/weights/best_detect.pt --image /home/uthira/pallet-detection/data/dataset/test/imagesmain/pallet_362.jpg

```

Script features:
- Real-time visualization of detection results
- Displays confidence scores for each detection
- Side-by-side view of original and annotated images
- GPU acceleration with half precision

### Segmentation Inference
Run single-image segmentation inference with live visualization:
```bash
python /home/uthira/pallet-detection/scripts/segmentation/inference/segmentation_inference_display.py --weights /home/uthira/pallet-detection/models/weights/best_segment.pt --image /home/uthira/pallet-detection/data/dataset/test/imagesmain/pallet_362.jpg

```

Script features:
- Real-time visualization of segmentation masks
- Different colors for each instance
- Side-by-side view of original and segmented images
- GPU acceleration with half precision

## Results

### Detection Results
```
- mAP@50: 0.59144
- Precision: 0.64713
- Recall: 0.51344
```
![Detection Results](results/aug_pallet_105.jpg)

### Segmentation Results
```
Mean IoU Overall: 0.0400
Max IoU Overall: 0.9516
Min IoU Overall: 0.0000
Median IoU: 0.0131
```
![Segmentation Results](results/vis_pallet_381.png)


### ros2 Package
```
/ros2_package/pallet
```

# ROS Pallet Package

This package provides functionality for publishing and processing images in ROS2, with support for both test images and live camera feed.

## Prerequisites

- ROS2 Humble
- Test images in JPG format (for testing mode)

## Installation

1. Source the ROS environment:
```bash
source /opt/ros/humble/setup.bash
```

2. Create a ROS workspace and navigate to it:
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
```

3. Copy the pallet package into the src folder:
```bash
cp -r /path/to/pallet src/
```

4. Build the package:
```bash
colcon build
```

5. Source the setup files:
```bash
source ./install/setup.bash
```

## Configuration

Before running the test image publisher, modify the image path in:
```
pallet/publishImage.py
```
Set the path to point to your folder containing test images (JPG format).

## Usage

### Publishing Test Images

To publish test images at 1Hz (simulating camera feed):
```bash
ros2 run pallet publish_test_images
```

Note: When using with real hardware, the image feed will come from an actual camera instead.

### Running Inference

In a new terminal, first source the ROS environment:
```bash
source /opt/ros/humble/setup.bash
source ./install/setup.bash
```

Then run the inference node:
```bash
ros2 run pallet infer
```




## Citation
If you use this project in your research, please cite:
```bibtex
@software{pallet_detection_2024,
author = {Uthiralakshmi Sivaraman},
title = {Pallet Detection and Segmentation},
year = {2024},
publisher = {GitHub},
url = {https://github.com/UthiraS/pallet_vision}
}
```

## Acknowledgments
- SAM2 by Meta AI Research 
- YOLOv11
- Roboflow for the initial dataset annotation