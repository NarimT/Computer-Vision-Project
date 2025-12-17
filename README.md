# Oil Palm Tree Detection from Drone Imagery

A deep learning project for detecting and counting oil palm trees in aerial drone imagery using state-of-the-art object detection models.

## Overview

This project implements and compares two object detection approaches for automated oil palm tree counting:
- **YOLOv8** (You Only Look Once v8)
- **Faster R-CNN** with ResNet50/ResNet101 backbones

The models are trained on high-resolution drone imagery that has been tiled into 512x512 pixel images for efficient processing.

**Paper:**
[Oil Palm Tree Detection from Drone Imagery (PDF)](st124895_st125983-.pdf)

## Results

### Best Model Performance

| Model | Backbone | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Faster R-CNN** | ResNet101 | **0.812** | 0.554 | **0.812** | 0.734 | 0.771 |
| YOLOv8 | YOLOv8-L | 0.811 | **0.578** | 0.643 | **0.901** | **0.751** |

- **Total experiments conducted:** 69 (6 YOLOv8 + 63 Faster R-CNN)
- **Best mAP@0.5:** 0.812 (Faster R-CNN ResNet101)
- **Best mAP@0.5:0.95:** 0.578 (YOLOv8-L)

### Sample Detection Output

<img src="cv_project/04_experiments/results/deteccion_fasterrcnn.png" width="800">

*Faster R-CNN detection results showing 8,083 detected palm trees*

Detection results are exported as GeoJSON files for GIS integration:
- `detected_palms_fasterrcnn.geojson`
- `detected_palms_yolo.geojson`

### Interactive Map

<img src="cv_project/04_experiments/results/Screenshot 2025-12-04 at 05.05.28.png" width="800">
 
*Inference results in Google Earth Engine*

🔗 **Direct Link:** (https://code.earthengine.google.com/0cd971d3d886ab24f4277f910cc730c3)

The interactive map allows you to:
- View all detected palm trees
- Toggle between YOLO and Faster R-CNN detections
- Visualize detection confidence levels
- Analyze tree health using NDVI
- Explore zone-by-zone statistics

## Project Structure

```
cv_project/
├── 01_raw_data/              # Original drone imagery
├── 02_final_tiles_only/      # Generated 512x512 image tiles
├── 03_datasets/              # Processed datasets for training
│   ├── oil_palm_coco_v1/     # COCO format dataset (v1)
│   ├── oil_palm_coco_v2/     # COCO format dataset (v2 - with test split)
│   ├── oil_palm_coco_v3/     # COCO format dataset (v3)
│   ├── oil_palm_yolo_v1/     # YOLO format dataset (v1)
│   └── oil_palm_yolo_v2/     # YOLO format dataset (v2)
├── 04_experiments/           # Experiment results and logs
│   ├── 01_yolov8/            # YOLOv8 experiment outputs
│   ├── 02_faster_rcnn/       # Faster R-CNN experiment outputs
│   ├── results/              # Final models and predictions
│   │   ├── yolo_best.pt      # Best YOLOv8 model weights
│   │   ├── restnet101__exp63_best.pt  # Best Faster R-CNN weights
│   │   ├── detected_palms_*.geojson   # Detection results
│   │   └── detection_*.png   # Visualization outputs
│   └── all_experiments_log.csv  # Complete experiment metrics
├── 05_notebooks/             # Training notebooks
│   ├── 01_YOLO_oil_palm_detection.ipynb
│   ├── 02_resnet50_oil_palm_detection_FasterRCNN.ipynb
│   └── 03_resnet101_oil_palm_detection_FasterRCNN.ipynb
│   └── 04_04_inference_best_models.ipynb
├── 06_utilities/             # Helper scripts
│   └── 00_create_tiles.ipynb # Tile generation script
```

## Dataset

### Statistics

**Dataset v1** (used for initial experiments):

| Split | Images | Annotations | Avg Objects/Image |
|-------|--------|-------------|-------------------|
| Train | 52 | 2,305 | 44.3 |
| Valid | 13 | 563 | 43.3 |
| **Total** | **65** | **2,868** | **44.1** |

**Dataset v2** (expanded dataset with test split):

| Split | Images | Annotations | Avg Objects/Image |
|-------|--------|-------------|-------------------|
| Train | 306 | 39,465 | 129.0 |
| Valid | 153 | 19,666 | 128.5 |
| Test | 51 | 6,817 | 133.7 |
| **Total** | **510** | **65,948** | **129.3** |

### Format
- **Image size:** 512 x 512 pixels
- **Annotation format:** COCO JSON (for Faster R-CNN) and YOLO TXT (for YOLOv8)
- **Class:** Single class - `palm-tree`

## Installation

### Requirements

```bash
# Clone the repository
git clone https://github.com/yourusername/oil-palm-detection.git
cd oil-palm-detection

# Create virtual environment (using uv)
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install torch torchvision
uv pip install ultralytics  # For YOLOv8
uv pip install albumentations pycocotools pandas matplotlib seaborn
```

### Dependencies

- Python 3.10+
- PyTorch 2.0+
- torchvision
- ultralytics (YOLOv8)
- albumentations
- pycocotools
- pandas
- matplotlib

## Usage

### Training YOLOv8

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8m.pt')

# Train
results = model.train(
    data='path/to/oil_palm_yolo_v1/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

### Training Faster R-CNN

See `05_notebooks/02_resnet50_oil_palm_detection_FasterRCNN.ipynb` or use the training script:

```python
# Configure experiment
exp_config = {
    'name': 'resnet101_experiment',
    'backbone': 'resnet101',  # or 'resnet50'
    'pretrained': True,
    'num_epochs': 110,
    'batch_size': 4,
    'learning_rate': 0.004,
    # ... see full config in training script
}
```

### Configurable Architecture Parameters (Faster R-CNN)

```python
{
    'box_score_thresh': 0.10,      # Detection confidence threshold
    'box_nms_thresh': 0.5,         # NMS IoU threshold
    'box_detections_per_img': 100, # Max detections per image
    'rpn_fg_iou_thresh': 0.7,      # RPN foreground threshold
    'rpn_bg_iou_thresh': 0.3,      # RPN background threshold
    'box_positive_fraction': 0.30  # Positive sample ratio
}
```

## Experiments

All 69 experiments are logged in `04_experiments/all_experiments_log.csv` with the following metrics:
- mAP@0.5, mAP@0.5:0.95, mAP@0.75
- Precision, Recall, F1-score
- Inference time (ms)
- Training time (hours)
- All hyperparameters and augmentation settings

## Key Findings

1. **Faster R-CNN ResNet101** achieved the best mAP@0.5 (0.812) and precision (0.812)
2. **YOLOv8-L** achieved the best mAP@0.5:0.95 (0.578) and recall (0.901)
3. Both models achieve similar mAP@0.5 (~0.81), but differ in precision/recall trade-off:
   - Faster R-CNN: Higher precision, lower recall
   - YOLOv8: Lower precision, higher recall
4. Dense object detection (129 objects/image avg) benefits from:
   - Higher `box_positive_fraction` (0.30 vs default 0.25)
   - Score threshold of 0.10 for better precision-recall balance
5. Default anchor aspect ratios [0.5, 1.0, 2.0] outperformed "optimized" ratios

## Authors
- **Nariman Tursaliev** (st125983)
- **Luis Medina** (st124895)


## License

This project is part of the Computer Vision course at AIT (Asian Institute of Technology).

## Acknowledgments

- Asian Institute of Technology (AIT)
