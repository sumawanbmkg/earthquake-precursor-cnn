# Pre-trained Models

## Available Models

### 1. VGG16 Multi-Task Model
- **File**: `best_vgg16_model.pth`
- **Size**: 1,256 MB (~1.2 GB)
- **Framework**: PyTorch
- **Architecture**: VGG16 + Multi-Task Head
- **Performance**:
  - Magnitude Accuracy: **98.68%**
  - Azimuth Accuracy: **54.93%**
  - Normal Detection: **100%**

### 2. EfficientNet-B0 Multi-Task Model (Recommended)
- **File**: `best_efficientnet_smote_model.pth`
- **Size**: 54 MB
- **Framework**: PyTorch
- **Architecture**: EfficientNet-B0 + Multi-Task Head
- **Performance**:
  - Magnitude Accuracy: **94.37%**
  - Azimuth Accuracy: **57.39%**
  - Inference Speed: **2.5x faster than VGG16**

## Class Mappings

### Magnitude Classes
| Index | Class | Description |
|-------|-------|-------------|
| 0 | Normal | No earthquake precursor |
| 1 | Moderate | M4.0 - M4.9 |
| 2 | Medium | M5.0 - M5.9 |
| 3 | Large | M6.0+ |

### Azimuth Classes
| Index | Class | Direction |
|-------|-------|-----------|
| 0 | Normal | No earthquake |
| 1 | N | North (337.5 - 22.5) |
| 2 | NE | Northeast (22.5 - 67.5) |
| 3 | E | East (67.5 - 112.5) |
| 4 | SE | Southeast (112.5 - 157.5) |
| 5 | S | South (157.5 - 202.5) |
| 6 | SW | Southwest (202.5 - 247.5) |
| 7 | W | West (247.5 - 292.5) |
| 8 | NW | Northwest (292.5 - 337.5) |

## Usage

### Load EfficientNet Model (Recommended)
```python
import torch
from src.models.efficientnet_multitask import EfficientNetMultiTask

# Load model
model = EfficientNetMultiTask(num_magnitude_classes=4, num_azimuth_classes=9)
model.load_state_dict(torch.load('models/best_efficientnet_smote_model.pth'))
model.eval()

# Inference
with torch.no_grad():
    mag_output, azi_output = model(input_tensor)
```

### Load VGG16 Model
```python
import torch
from src.models.vgg16_multitask import VGG16MultiTask

# Load model
model = VGG16MultiTask(num_magnitude_classes=4, num_azimuth_classes=9)
model.load_state_dict(torch.load('models/best_vgg16_model.pth'))
model.eval()
```

## Model Selection Guide

| Use Case | Recommended Model | Reason |
|----------|-------------------|--------|
| Production | EfficientNet-B0 | Smaller, faster, better azimuth |
| Mobile/Edge | EfficientNet-B0 | Only 54 MB |
| Maximum Accuracy | VGG16 | 98.68% magnitude |
| Research | Both | Compare results |

## Download from GitHub Releases

If models are not included in the repository (due to size), download from:
- [GitHub Releases](https://github.com/sumawanbmkg/earthquake-precursor-cnn/releases)

## Hyperparameters

### EfficientNet-B0 (Tuned with Optuna)
```json
{
  "learning_rate": 0.000989,
  "dropout": 0.444,
  "weight_decay": 0.000052,
  "batch_size": 32,
  "epochs": 12,
  "best_epoch": 5
}
```

### VGG16
```json
{
  "learning_rate": 0.0001,
  "dropout": 0.5,
  "weight_decay": 0.0001,
  "batch_size": 32,
  "epochs": 11
}
```

