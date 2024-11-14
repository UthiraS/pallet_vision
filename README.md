# Pallet Detection and Segmentation
This project implements an end-to-end solution for pallet detection and segmentation using YOLOv11 and SAM2 (Segment Anything Model 2). It includes tools for annotation, training, inference, and evaluation.

![Pallet Detection Example](results/aug_pallet_287.jpg)
![Pallet Segmentation Example](results/vis_pallet_337.png)

## Dataset
The complete dataset is available on Google Drive:
- [Dataset](https://drive.google.com/drive/u/1/folders/1OIxiRhRfp1iDgKj_7L7DgIXOJ6OHqHTa)


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
python detection_inference_display.py --weights path/to/best.pt --image path/to/image.jpg
```

Script features:
- Real-time visualization of detection results
- Displays confidence scores for each detection
- Side-by-side view of original and annotated images
- GPU acceleration with half precision

### Segmentation Inference
Run single-image segmentation inference with live visualization:
```bash
python segmentation_inference_display.py --weights path/to/best.pt --image path/to/image.jpg
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

## Requirements
```txt
torch>=2.0.0
ultralytics>=8.0.0
segment-anything-2==0.1.0
supervision>=0.3.0
opencv-python>=4.8.0
numpy>=1.24.0
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