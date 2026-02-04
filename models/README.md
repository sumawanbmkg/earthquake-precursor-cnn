# Pre-trained Models

This directory contains pre-trained models for earthquake precursor detection.

## Available Models

| Model | File | Size | Magnitude Acc | Azimuth Acc | Download |
|-------|------|------|---------------|-------------|----------|
| **VGG16** | `best_vgg16_model.pth` | 1.26 GB | 98.68% | 54.93% | [Download](https://github.com/sumawanbmkg/earthquake-precursor-cnn/releases/download/v1.0.0/best_vgg16_model.pth) |
| **EfficientNet-B0** | `best_efficientnet_smote_model.pth` | 54 MB | 94.37% | 57.39% | [Download](https://github.com/sumawanbmkg/earthquake-precursor-cnn/releases/download/v1.0.0/best_efficientnet_smote_model.pth) |

## Quick Download

### Option 1: Using Python Script

```bash
python scripts/download_models.py
```

### Option 2: Using wget/curl

```bash
# Download EfficientNet-B0 (recommended, smaller)
wget https://github.com/sumawanbmkg/earthquake-precursor-cnn/releases/download/v1.0.0/best_efficientnet_smote_model.pth -O models/best_efficientnet_smote_model.pth

# Download VGG16 (larger, higher magnitude accuracy)
wget https://github.com/sumawanbmkg/earthquake-precursor-cnn/releases/download/v1.0.0/best_vgg16_model.pth -O models/best_vgg16_model.pth
```

### Option 3: Manual Download

1. Go to [Releases](https://github.com/sumawanbmkg/earthquake-precursor-cnn/releases)
2. Download the model files from the latest release
3. Place them in this `models/` directory

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
