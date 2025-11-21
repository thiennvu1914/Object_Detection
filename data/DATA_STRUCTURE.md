# Dataset Structure

This directory contains the training images and reference images for food detection and classification.

⚠️ **Note:** Image files are **NOT** included in the Git repository. You need to prepare your own dataset.

## Directory Structure

```
data/
├── images/              # Training images (25 images)
│   ├── .gitkeep
│   ├── image_01.jpg
│   ├── image_02.jpg
│   └── ...
│
├── labels/              # YOLO format labels (25 files)
│   ├── .gitkeep
│   ├── image_01.txt    # Format: class x_center y_center width height
│   ├── image_02.txt
│   └── ...
│
└── ref_images/          # Reference images for classification (30 images)
    ├── .gitkeep
    ├── coconut/         # 6 images
    │   ├── ref_001.jpg
    │   └── ...
    ├── cua/             # 6 images
    ├── macaron/         # 6 images
    ├── meden/           # 6 images
    └── melon/           # 6 images
```

## Training Images (`images/`)

**Purpose:** Training object detection model

**Format:** 
- JPG/PNG images
- Recommended size: 640x640 or similar aspect ratio
- Total: 25 images

**Naming Convention:**
```
image_01.jpg, image_02.jpg, ..., image_25.jpg
```

## Labels (`labels/`)

**Purpose:** YOLO format annotations for training

**Format:** One text file per image
```
class x_center y_center width height
```

**Example:** `image_01.txt`
```
0 0.5234 0.4123 0.1234 0.0987
0 0.3456 0.6789 0.0876 0.1234
```

**Coordinates:**
- All values normalized to [0, 1]
- x_center, y_center: Center of bounding box
- width, height: Box dimensions
- class: All set to `0` (single-class detection)

**Generate Labels:**
Use labeling tools like:
- [LabelImg](https://github.com/HumanSignal/labelImg)
- [Roboflow](https://roboflow.com/)
- [CVAT](https://github.com/opencv/cvat)

## Reference Images (`ref_images/`)

**Purpose:** Classification via embedding similarity matching

**Structure:** One folder per class, 6 reference images each

**Classes:**
1. **coconut** - Coconut pieces
2. **cua** - Crab/seafood items
3. **macaron** - Small round macarons
4. **meden** - Meden food items
5. **melon** - Melon pieces

**Requirements:**
- Clear, well-lit images
- Focused on single object
- Diverse angles and lighting
- Similar to objects you want to detect

**Generate Reference Images:**
```bash
# Interactive tool to label detected objects
python src/generate_ref_images.py

# Press 1-5 to label objects
# 1: coconut, 2: cua, 3: macaron, 4: meden, 5: melon
```

## Prepare Your Own Dataset

### Option 1: Collect Images

1. **Capture images** of food items in trays
2. **Annotate** with YOLO format using labeling tool
3. **Split** into training images (80-90%) and validation
4. **Place** in `data/images/` and `data/labels/`

### Option 2: Use Existing Dataset

If you have COCO or other format:

```bash
# Convert to YOLO format
python src/prepare_dataset.py --input <your_dataset> --output data/
```

## Dataset Statistics

Current dataset (not included):
- **Training images:** 25 images
- **Labels:** 25 text files (all class 0)
- **Reference images:** 30 images (6 per class)
- **Train/Val split:** 90/10

**Image sizes:** Mixed (auto-resized to 640x640 during training)

## Data Augmentation

During training, the following augmentations are applied:
- Mosaic: 1.0
- MixUp: 0.1
- HSV color jitter
- Random flips
- Scale and translate

See `src/train_yolo.py` for full augmentation config.

## .gitignore Rules

The following files are excluded from Git:
```gitignore
data/images/*.jpg
data/images/*.png
data/labels/*.txt
data/ref_images/**/*.jpg
data/ref_images/**/*.png
```

Only `.gitkeep` files are tracked to preserve directory structure.

## Example Dataset Layout

After preparing your dataset:

```
data/
├── DATA_STRUCTURE.md   # This file
├── images/             # 25 training images (~30MB)
│   ├── .gitkeep
│   ├── image_01.jpg
│   └── ...
├── labels/             # 25 label files
│   ├── .gitkeep
│   ├── image_01.txt    # "0 0.523 0.412 0.123 0.098"
│   └── ...
└── ref_images/         # 30 reference images (~10MB)
    ├── coconut/
    │   ├── ref_001.jpg
    │   └── ... (6 images)
    ├── cua/
    ├── macaron/
    ├── meden/
    └── melon/
```

## Verification

Check your dataset:

```powershell
# Count images
Get-ChildItem data\images -File | Measure-Object | Select Count

# Count labels
Get-ChildItem data\labels -File -Filter "*.txt" | Measure-Object | Select Count

# Count reference images per class
Get-ChildItem data\ref_images -Recurse -File | Group-Object Directory | Select Name, Count
```

Expected output:
```
Images: 25
Labels: 25
coconut: 6
cua: 6
macaron: 6
meden: 6
melon: 6
```
