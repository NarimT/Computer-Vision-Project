"""
Script to prepare Oil Palm dataset in different formats
Supports: YOLO, COCO
Organizes images and labels in train/val structure
"""
import shutil
import random
import json
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Project paths
    'paths': {
        'project_root': Path("/Users/lujem/Documents/AIT/03_semester/computer_vision/cv_project"),
        'images_source': "02_final_tiles_only",
        'datasets_base': "03_datasets"
    },

    # Dataset information
    'dataset': {
        'version': 'v1',
        'num_classes': 1,
        'class_names': ['oil_palm'],
        'category_id': 0,  # For COCO
        'train_ratio': 0.8  # 80% train, 20% val
    },

    # Format configurations - each format has its own labels source
    'formats': {
        'yolo': {
            'name': 'oil_palm_yolo_v1',
            'labels_source': "label_studio_exports/YOLO_FORMAT/project-2-at-2025-11-17-08-28-50c8b0d7/labels",  # YOLO format labels
            'folders': ['images/train', 'images/val', 'labels/train', 'labels/val']
        },
        'coco': {
            'name': 'oil_palm_coco_v1',
            'labels_source': "label_studio_exports/COCO_FORMAT/project-2-at-2025-11-18-14-50-50c8b0d7",  # COCO format labels
            'folders': ['train', 'val', 'annotations']
        }
    },

    # Google Drive paths (for config files)
    'google_drive': {
        'datasets_path': '/content/drive/MyDrive/cv_project/03_datasets'
    },

    # Dataset info (for COCO metadata)
    'coco_info': {
        'description': 'Oil Palm Detection Dataset',
        'version': 'v1',
        'year': 2025,
        'contributor': 'Luis M',
        'date_created': '2025-11-17'
    }
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_absolute_path(relative_path: str) -> Path:
    """Convert relative path to absolute using project_root"""
    return CONFIG['paths']['project_root'] / relative_path


def get_dataset_root(format_type: str) -> Path:
    """Return dataset path based on format"""
    format_config = CONFIG['formats'].get(format_type.lower())
    if not format_config:
        raise ValueError(f"Unsupported format: {format_type}")

    return get_absolute_path(CONFIG['paths']['datasets_base']) / format_config['name']


def get_labels_source(format_type: str) -> Path:
    """Return labels source path based on format"""
    format_config = CONFIG['formats'].get(format_type.lower())
    if not format_config:
        raise ValueError(f"Unsupported format: {format_type}")

    return get_absolute_path(format_config['labels_source'])


def clean_dataset_folder(dataset_root: Path):
    """Clean dataset folder to avoid duplicates"""
    if dataset_root.exists():
        print(f"[CLEAN] Cleaning existing folder: {dataset_root.name}")

        folders_to_clean = [
            dataset_root / "images",
            dataset_root / "labels",
            dataset_root / "annotations"
        ]

        for folder in folders_to_clean:
            if folder.exists():
                shutil.rmtree(folder)
                print(f"  [OK] Removed: {folder.name}/")

        print("[OK] Folder cleaned\n")
    else:
        print(f"[INFO] Folder does not exist, will create new: {dataset_root.name}\n")


def create_folder_structure(dataset_root: Path, format_type: str):
    """Create folder structure based on format"""
    format_config = CONFIG['formats'][format_type]

    print(f"[{format_type.upper()}] Creating folder structure...")

    for folder_path in format_config['folders']:
        full_path = dataset_root / folder_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] {full_path.relative_to(dataset_root.parent)}")

    print(f"[OK] {format_type.upper()} structure ready\n")


# ============================================================================
# CONFIGURATION FILE FUNCTIONS
# ============================================================================

def create_yolo_config(dataset_root: Path) -> Tuple[Path, Path]:
    """Create data.yaml files for YOLO"""
    dataset_config = CONFIG['dataset']
    drive_config = CONFIG['google_drive']
    format_config = CONFIG['formats']['yolo']

    # data.yaml for local use
    local_yaml_content = f"""# YOLOv8 Dataset Configuration - Local Paths
# Format: YOLO
path: {dataset_root}
train: images/train
val: images/val

nc: {dataset_config['num_classes']}
names: {dataset_config['class_names']}
"""

    local_yaml_path = dataset_root / "data.yaml"
    with open(local_yaml_path, 'w') as f:
        f.write(local_yaml_content)
    print(f"[SAVE] Created: data.yaml (local paths)")

    # data_colab.yaml for Google Colab use
    colab_yaml_content = f"""# YOLOv8 Dataset Configuration - Google Colab Paths
# Format: YOLO
path: {drive_config['datasets_path']}/{format_config['name']}
train: images/train
val: images/val

nc: {dataset_config['num_classes']}
names: {dataset_config['class_names']}
"""

    colab_yaml_path = dataset_root / "data_colab.yaml"
    with open(colab_yaml_path, 'w') as f:
        f.write(colab_yaml_content)
    print(f"[SAVE] Created: data_colab.yaml (Google Drive paths)")

    return local_yaml_path, colab_yaml_path


def create_coco_config(dataset_root: Path) -> Path:
    """Create configuration file for COCO (informational only)"""
    dataset_config = CONFIG['dataset']
    drive_config = CONFIG['google_drive']
    format_config = CONFIG['formats']['coco']

    config_content = f"""# COCO Dataset Configuration
# Format: COCO
dataset_name: {format_config['name']}
num_classes: {dataset_config['num_classes']}
class_names: {dataset_config['class_names']}

# Local paths
train_images: {dataset_root}/train
val_images: {dataset_root}/val
train_annotations: {dataset_root}/annotations/instances_train.json
val_annotations: {dataset_root}/annotations/instances_val.json

# Google Colab paths (for Faster R-CNN script)
colab_train_images: {drive_config['datasets_path']}/{format_config['name']}/train
colab_val_images: {drive_config['datasets_path']}/{format_config['name']}/val
colab_train_annotations: {drive_config['datasets_path']}/{format_config['name']}/annotations/instances_train.json
colab_val_annotations: {drive_config['datasets_path']}/{format_config['name']}/annotations/instances_val.json

# Usage in Faster R-CNN script:
# train_images_dir = Path('/content/drive/MyDrive/cv_project/03_datasets/{format_config['name']}/train')
# val_images_dir = Path('/content/drive/MyDrive/cv_project/03_datasets/{format_config['name']}/val')
# train_ann_file = Path('/content/drive/MyDrive/cv_project/03_datasets/{format_config['name']}/annotations/instances_train.json')
# val_ann_file = Path('/content/drive/MyDrive/cv_project/03_datasets/{format_config['name']}/annotations/instances_val.json')
"""

    config_path = dataset_root / "dataset_info.txt"
    with open(config_path, 'w') as f:
        f.write(config_content)
    print(f"[SAVE] Created: dataset_info.txt (informational)")

    return config_path


# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def get_image_label_pairs_yolo(format_type: str) -> List[Dict]:
    """Get valid image-label pairs for YOLO format"""
    labels_source = get_labels_source(format_type)
    images_source = get_absolute_path(CONFIG['paths']['images_source'])

    label_files = list(labels_source.glob("*.txt"))
    print(f"[INFO] Total YOLO label files found: {len(label_files)}")
    print(f"[INFO] Labels source: {labels_source}")

    image_label_pairs = []
    for label_file in label_files:
        # Extract tile name from label filename
        # Format: UUID-tile_0000_XXXX.txt -> tile_0000_XXXX.png
        label_name = label_file.name
        tile_name = label_name.split('-', 1)[1].replace('.txt', '.png')

        image_path = images_source / tile_name

        if image_path.exists():
            image_label_pairs.append({
                'image': image_path,
                'label': label_file,
                'tile_name': tile_name
            })
            print(f"  [OK] Found: {tile_name}")
        else:
            print(f"  [WARNING] Not found: {tile_name}")

    print(f"\n[INFO] Total valid image-label pairs: {len(image_label_pairs)}\n")
    return image_label_pairs


def get_image_label_pairs_coco(format_type: str) -> List[Dict]:
    """
    Get valid image-label pairs for COCO format.
    Returns list of dicts with image info and annotation data.
    """
    labels_source = get_labels_source(format_type)
    images_source = get_absolute_path(CONFIG['paths']['images_source'])

    # Check if COCO labels source exists
    if not labels_source.exists():
        print(f"[ERROR] COCO labels source not found: {labels_source}")
        print(f"[INFO] Please export your annotations in COCO format from Label Studio")
        print(f"[INFO] Expected location: {labels_source}")
        return []

    # Check for annotation file
    annotation_file = labels_source / "result.json"
    if not annotation_file.exists():
        print(f"[ERROR] COCO annotation file not found: {annotation_file}")
        print(f"[INFO] Please ensure you have a 'result.json' file in the COCO export folder")
        return []

    print(f"[COCO] Loading COCO annotations from: {annotation_file}")

    # Load COCO JSON
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)

    print(f"[COCO] Found {len(coco_data['images'])} images")
    print(f"[COCO] Found {len(coco_data['annotations'])} annotations")
    print(f"[COCO] Found {len(coco_data['categories'])} categories\n")

    # Create mapping of image_id to annotations
    image_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_to_anns:
            image_to_anns[img_id] = []
        image_to_anns[img_id].append(ann)

    # Create image-label pairs
    pairs = []
    matched = 0
    skipped = 0

    for img_info in coco_data['images']:
        img_id = img_info['id']
        file_name_original = img_info['file_name']

        # Extract actual tile name from Label Studio path
        # Format: ../../label-studio/.../UUID-tile_0000_0000.png -> tile_0000_0000.png
        if '/' in file_name_original:
            # Get just the filename from the path
            file_name_with_uuid = file_name_original.split('/')[-1]
        else:
            file_name_with_uuid = file_name_original

        # Remove UUID prefix if present: UUID-tile_0000_0000.png -> tile_0000_0000.png
        if '-' in file_name_with_uuid:
            file_name = file_name_with_uuid.split('-', 1)[1]
        else:
            file_name = file_name_with_uuid

        # Find corresponding image in images_source
        image_path = images_source / file_name

        if not image_path.exists():
            print(f"[WARNING] Image not found: {file_name}")
            skipped += 1
            continue

        # Get annotations for this image
        anns = image_to_anns.get(img_id, [])

        if len(anns) == 0:
            print(f"[WARNING] No annotations for: {file_name}")
            skipped += 1
            continue

        # Store pair with COCO data (use clean file_name, not original)
        # Update image_info to use clean filename
        img_info_clean = img_info.copy()
        img_info_clean['file_name'] = file_name

        pairs.append({
            'tile_name': file_name,
            'image': image_path,
            'image_info': img_info_clean,
            'annotations': anns
        })
        matched += 1

    print(f"\n[MATCH] Matched {matched} image-annotation pairs")
    if skipped > 0:
        print(f"[SKIP] Skipped {skipped} images (not found or no annotations)")

    return pairs


def split_dataset(image_label_pairs: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Split dataset into train/val"""
    train_ratio = CONFIG['dataset']['train_ratio']

    random.seed(42)  # For reproducibility
    random.shuffle(image_label_pairs)

    split_idx = int(len(image_label_pairs) * train_ratio)
    train_pairs = image_label_pairs[:split_idx]
    val_pairs = image_label_pairs[split_idx:]

    total = len(image_label_pairs)
    print(f"[SPLIT] Dataset split:")
    print(f"  Train: {len(train_pairs)} images ({len(train_pairs)/total*100:.1f}%)")
    print(f"  Val:   {len(val_pairs)} images ({len(val_pairs)/total*100:.1f}%)\n")

    return train_pairs, val_pairs


# ============================================================================
# DATASET PREPARATION BY FORMAT
# ============================================================================

def prepare_yolo_dataset(dataset_root: Path, train_pairs: List[Dict], val_pairs: List[Dict]):
    """Prepare dataset in YOLO format"""
    print("[YOLO] Preparing dataset in YOLO format...\n")

    train_images = dataset_root / "images" / "train"
    val_images = dataset_root / "images" / "val"
    train_labels = dataset_root / "labels" / "train"
    val_labels = dataset_root / "labels" / "val"

    # Copy files to train
    print("[COPY] Copying training files...")
    for pair in train_pairs:
        shutil.copy2(pair['image'], train_images / pair['tile_name'])
        label_dest = train_labels / pair['tile_name'].replace('.png', '.txt')
        shutil.copy2(pair['label'], label_dest)
        print(f"  [OK] Train: {pair['tile_name']}")

    # Copy files to val
    print("\n[COPY] Copying validation files...")
    for pair in val_pairs:
        shutil.copy2(pair['image'], val_images / pair['tile_name'])
        label_dest = val_labels / pair['tile_name'].replace('.png', '.txt')
        shutil.copy2(pair['label'], label_dest)
        print(f"  [OK] Val:   {pair['tile_name']}")

    # Statistics
    print(f"\n[STATS] YOLO structure created:")
    print(f"  images/train/: {len(list(train_images.glob('*.png')))} images")
    print(f"  images/val/:   {len(list(val_images.glob('*.png')))} images")
    print(f"  labels/train/: {len(list(train_labels.glob('*.txt')))} labels")
    print(f"  labels/val/:   {len(list(val_labels.glob('*.txt')))} labels")

    # Bounding box statistics
    total_boxes_train = sum(len(open(f).readlines()) for f in train_labels.glob('*.txt'))
    total_boxes_val = sum(len(open(f).readlines()) for f in val_labels.glob('*.txt'))

    print(f"\n[STATS] Bounding boxes:")
    print(f"  Train: {total_boxes_train} boxes ({total_boxes_train/len(train_pairs):.1f} per image)")
    print(f"  Val:   {total_boxes_val} boxes ({total_boxes_val/len(val_pairs):.1f} per image)")
    print(f"  Total: {total_boxes_train + total_boxes_val} boxes\n")


def prepare_coco_dataset(dataset_root: Path, train_pairs: List[Dict], val_pairs: List[Dict]):
    """
    Prepare dataset in COCO format.
    Creates train/val folders with images and COCO annotation JSON files.
    """
    print("[COCO] Preparing dataset in COCO format...\n")

    # Create directories
    train_images_dir = dataset_root / "train"
    val_images_dir = dataset_root / "val"
    annotations_dir = dataset_root / "annotations"

    train_images_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # Helper function to create COCO JSON
    def create_coco_json(pairs: List[Dict], split_name: str):
        """Create COCO format JSON for train or val split"""
        coco_output = {
            'info': CONFIG['coco_info'],
            'licenses': [],
            'categories': [
                {
                    'id': 1,  # COCO categories start at 1
                    'name': CONFIG['dataset']['class_names'][0],
                    'supercategory': 'plant'
                }
            ],
            'images': [],
            'annotations': []
        }

        annotation_id = 1

        for idx, pair in enumerate(pairs):
            # Add image info (with new sequential ID)
            image_info = {
                'id': idx + 1,  # Sequential ID starting from 1
                'file_name': pair['tile_name'],
                'width': pair['image_info']['width'],
                'height': pair['image_info']['height']
            }
            coco_output['images'].append(image_info)

            # Add annotations for this image
            for ann in pair['annotations']:
                annotation = {
                    'id': annotation_id,
                    'image_id': idx + 1,  # Reference to new image ID
                    'category_id': 1,  # oil_palm category
                    'bbox': ann['bbox'],  # [x, y, width, height]
                    'area': ann['area'],
                    'iscrowd': 0,
                    'segmentation': []
                }
                coco_output['annotations'].append(annotation)
                annotation_id += 1

        # Save JSON file
        json_file = annotations_dir / f'instances_{split_name}.json'
        with open(json_file, 'w') as f:
            json.dump(coco_output, f, indent=2)

        return json_file, len(coco_output['images']), len(coco_output['annotations'])

    # Copy train images and create train JSON
    print("[COPY] Copying training images...")
    for pair in train_pairs:
        dest = train_images_dir / pair['tile_name']
        shutil.copy2(pair['image'], dest)
        print(f"  [OK] Train: {pair['tile_name']}")

    train_json, train_imgs, train_anns = create_coco_json(train_pairs, 'train')
    print(f"\n[OK] Created: {train_json.name}")
    print(f"  Images: {train_imgs}")
    print(f"  Annotations: {train_anns}")

    # Copy val images and create val JSON
    print("\n[COPY] Copying validation images...")
    for pair in val_pairs:
        dest = val_images_dir / pair['tile_name']
        shutil.copy2(pair['image'], dest)
        print(f"  [OK] Val:   {pair['tile_name']}")

    val_json, val_imgs, val_anns = create_coco_json(val_pairs, 'val')
    print(f"\n[OK] Created: {val_json.name}")
    print(f"  Images: {val_imgs}")
    print(f"  Annotations: {val_anns}")

    # Statistics
    print(f"\n[STATS] COCO structure created:")
    print(f"  train/: {len(list(train_images_dir.glob('*.png')))} images")
    print(f"  val/:   {len(list(val_images_dir.glob('*.png')))} images")
    print(f"  annotations/instances_train.json: {train_anns} annotations")
    print(f"  annotations/instances_val.json:   {val_anns} annotations")

    print(f"\n[STATS] Bounding boxes:")
    print(f"  Train: {train_anns} boxes ({train_anns/len(train_pairs):.1f} per image)")
    print(f"  Val:   {val_anns} boxes ({val_anns/len(val_pairs):.1f} per image)")
    print(f"  Total: {train_anns + val_anns} boxes\n")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def prepare_dataset(format_type: str = 'yolo'):
    """
    Prepare dataset in specified format

    Args:
        format_type: 'yolo' or 'coco'
    """
    format_type = format_type.lower()

    print("=" * 70)
    print(f"  Dataset Preparation - Format: {format_type.upper()}")
    print("=" * 70)
    print()

    # Validate format
    if format_type not in CONFIG['formats']:
        raise ValueError(f"Invalid format: {format_type}. Use: {list(CONFIG['formats'].keys())}")

    # Get format configuration
    format_config = CONFIG['formats'][format_type]
    dataset_root = get_dataset_root(format_type)
    dataset_name = format_config['name']

    print(f"[INFO] Dataset: {dataset_name}")
    print(f"[INFO] Format: {format_type.upper()}")
    print(f"[INFO] Output path: {dataset_root}")
    print(f"[INFO] Labels source: {get_labels_source(format_type)}\n")

    # Clean existing folder
    clean_dataset_folder(dataset_root)

    # Create folder structure based on format
    create_folder_structure(dataset_root, format_type)

    # Get image-label pairs based on format
    if format_type == 'yolo':
        image_label_pairs = get_image_label_pairs_yolo(format_type)
    elif format_type == 'coco':
        image_label_pairs = get_image_label_pairs_coco(format_type)
    else:
        raise ValueError(f"Unsupported format: {format_type}")

    if len(image_label_pairs) == 0:
        print("[ERROR] No valid image-label pairs found!")
        return False

    # Split into train/val
    train_pairs, val_pairs = split_dataset(image_label_pairs)

    # Prepare dataset based on format
    if format_type == 'yolo':
        prepare_yolo_dataset(dataset_root, train_pairs, val_pairs)
        create_yolo_config(dataset_root)
    elif format_type == 'coco':
        prepare_coco_dataset(dataset_root, train_pairs, val_pairs)
        create_coco_config(dataset_root)

    print("[OK] Dataset prepared successfully!")
    print(f"\n[NEXT STEP] Upload to Google Drive:")
    print(f"  Upload this folder: {dataset_root}")
    print(f"  To: My Drive/cv_project/03_datasets/{dataset_name}/")

    if format_type == 'yolo':
        print(f"  In Colab (YOLOv8): use data_colab.yaml")
    else:
        print(f"  In Colab (Faster R-CNN): use data_config.yaml")

    return True


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Prepare Oil Palm dataset in YOLO or COCO format'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['yolo', 'coco'],
        default='yolo',
        help='Dataset format: yolo or coco (default: yolo)'
    )

    args = parser.parse_args()

    success = prepare_dataset(format_type=args.format)

    if success:
        print("\n" + "=" * 70)
        print("  Dataset prepared successfully!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("  Error preparing dataset")
        print("=" * 70)
