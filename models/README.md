# Pre-trained Models

This directory contains pre-trained models for earthquake precursor detection.

## Available Models

| Model | File | Size | Parts | Magnitude Acc | Azimuth Acc |
|-------|------|------|-------|---------------|-------------|
| **VGG16** | `best_vgg16_model.pth` | 1.26 GB | 53 | 98.68% | 54.93% |
| **EfficientNet-B0** | `best_efficientnet_smote_model.pth` | 54 MB | 3 | 94.37% | 57.39% |

## Quick Download (Recommended)

The easiest way to download models:

```bash
python scripts/download_models.py
```

This will automatically:
1. Download all parts from GitHub Releases
2. Merge them into the complete model file
3. Verify the checksum

### Download Specific Model

```bash
# Download only EfficientNet (recommended, smaller)
python scripts/download_models.py --model efficientnet

# Download only VGG16 (larger, higher accuracy)
python scripts/download_models.py --model vgg16
```

## Manual Download

Due to GitHub's file size limits, models are split into parts (<25MB each).

1. Go to [Releases](https://github.com/sumawanbmkg/earthquake-precursor-cnn/releases/tag/v1.0.0)
2. Download all `.part*` files for your desired model
3. Use the merge script:

```bash
# For EfficientNet
python scripts/merge_model.py --manifest models/parts/efficientnet/best_efficientnet_smote_model.manifest.txt --output models/

# For VGG16
python scripts/merge_model.py --manifest models/parts/vgg16/best_vgg16_model.manifest.txt --output models/
```

## Checksums

| Model | MD5 |
|-------|-----|
| VGG16 | `5804d2b155e7787b95647b1ccb7ee9a6` |
| EfficientNet-B0 | `457549bb8d1e5f796787745430014edc` |

## Model Architecture

### VGG16 Multi-Task
- Base: VGG16 pretrained on ImageNet
- Modified for multi-task learning (magnitude + azimuth)
- Input: 224×224×3 RGB spectrogram
- Output: 4 magnitude classes + 9 azimuth classes

### EfficientNet-B0 Multi-Task
- Base: EfficientNet-B0 pretrained on ImageNet
- Modified for multi-task learning (magnitude + azimuth)
- Input: 224×224×3 RGB spectrogram
- Output: 4 magnitude classes + 9 azimuth classes
- **Recommended for production** (26× smaller, 2.5× faster)

## Class Mappings

### Magnitude Classes
```json
{
  "0": "Small (M4.0-4.9)",
  "1": "Medium (M5.0-5.9)",
  "2": "Large (M6.0-6.9)",
  "3": "Major (M7.0+)"
}
```

### Azimuth Classes
```json
{
  "0": "N (337.5-22.5°)",
  "1": "NE (22.5-67.5°)",
  "2": "E (67.5-112.5°)",
  "3": "SE (112.5-157.5°)",
  "4": "S (157.5-202.5°)",
  "5": "SW (202.5-247.5°)",
  "6": "W (247.5-292.5°)",
  "7": "NW (292.5-337.5°)",
  "8": "Normal (no precursor)"
}
```

## Usage Example

```python
import torch
from src.predictor import EarthquakePrecursorPredictor

# Load EfficientNet model (recommended)
predictor = EarthquakePrecursorPredictor(
    model_name='efficientnet',
    model_path='models/best_efficientnet_smote_model.pth'
)

# Make prediction
result = predictor.predict('path/to/spectrogram.png')
print(f"Magnitude: {result['magnitude']}")
print(f"Azimuth: {result['azimuth']}")
```

## Model Files in This Directory

- `README.md` - This file
- `efficientnet_class_mappings.json` - Class label mappings for EfficientNet
- `efficientnet_hyperparameters.json` - Training hyperparameters
- `vgg16_class_mappings.json` - Class label mappings for VGG16

## Notes

- Model files (`.pth`) are hosted on GitHub Releases due to size constraints
- Use the download script or manual download to obtain the models
- EfficientNet-B0 is recommended for most use cases (smaller, faster, better generalization)
- VGG16 achieves higher magnitude accuracy but requires more resources
