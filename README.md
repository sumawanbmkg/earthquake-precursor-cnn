# Earthquake Precursor Detection using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Submitted-yellow.svg)](docs/paper.pdf)
[![DOI](https://img.shields.io/badge/DOI-Pending-orange.svg)](https://doi.org/)

**A Comparative Study of VGG16 and EfficientNet-B0 for Geomagnetic Precursor Detection**

> Deep learning models for earthquake precursor detection from geomagnetic data using multi-task learning. Achieves 98.68% magnitude accuracy (VGG16) and 57.39% azimuth accuracy (EfficientNet-B0) with 26× smaller model size.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Results](#results)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Model Comparison](#model-comparison)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

##  Overview

This repository contains the implementation of our research on earthquake precursor detection using deep learning. We compare two state-of-the-art CNN architectures (VGG16 and EfficientNet-B0) for multi-task learning of earthquake magnitude and azimuth prediction from geomagnetic spectrogram data.

### Key Contributions

- **Comparative Study**: First comprehensive comparison of VGG16 vs EfficientNet-B0 for earthquake precursor detection
- **Multi-Task Learning**: Simultaneous prediction of magnitude (4 classes) and azimuth (9 classes)
- **Explainable AI**: Grad-CAM visualizations showing model attention on physically meaningful patterns
- **Production Ready**: EfficientNet-B0 model (20 MB) suitable for real-time deployment
- **Rigorous Validation**: Fixed split + LOEO validation to ensure true generalization

---

## ✨ Key Features

- 🎯 **High Accuracy**: 98.68% magnitude accuracy (VGG16), 57.39% azimuth accuracy (EfficientNet-B0)
-  **Efficient**: EfficientNet-B0 is 26 smaller and 2.5 faster than VGG16
-  **Interpretable**: Grad-CAM visualizations for model explainability
-  **Comprehensive**: Detailed comparison of architectures, performance, and efficiency
-  **Production Ready**: Optimized for deployment with automated hyperparameter tuning
-  **Well Validated**: Fixed split + LOEO cross-validation

---

##  Results

### Model Performance

| Model | Magnitude Acc | Azimuth Acc | Model Size | Inference Speed | Deployment |
|-------|---------------|-------------|------------|-----------------|------------|
| **VGG16** | **98.68%** | 54.93% | 528 MB | 125 ms | Cloud only |
| **EfficientNet-B0** | 94.37% | **57.39%** | **20 MB** | **50 ms** |  Mobile/Edge |

### Key Metrics

- **Dataset**: 256 unique earthquake events (2018-2025, Indonesia)
- **Total Samples**: 1,972 spectrograms (1,084 precursor + 888 normal)
- **Normal Detection**: 100% accuracy (both models)
- **Generalization**: LOEO validation confirms <5% accuracy drop
- **Explainability**: 100% prediction agreement between models on Grad-CAM samples

### Grad-CAM Visualization

Both models focus on physically meaningful patterns (ULF frequency bands 0.001-0.01 Hz):

![Grad-CAM Comparison](figures/gradcam_comparison.png)

---

##  Installation

### Prerequisites

- Python 3.10 or higher
- CUDA 11.8+ (optional, for GPU acceleration)
- 16 GB RAM minimum
- 50 GB free disk space

### Step 1: Clone Repository

\\\ash
git clone https://github.com/yourusername/earthquake-precursor-cnn.git
cd earthquake-precursor-cnn
\\\

### Step 2: Create Virtual Environment

\\\ash
# Using venv
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate
\\\

### Step 3: Install Dependencies

\\\ash
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
\\\

### Step 4: Download Pre-trained Models

\\\ash
# Download models from releases
python scripts/download_models.py

# Or manually download from:
# https://github.com/yourusername/earthquake-precursor-cnn/releases
\\\

### Step 5: Verify Installation

\\\ash
python scripts/verify_installation.py
\\\

Expected output:
\\\
 Python version: 3.10.x
 PyTorch version: 2.0.x
 CUDA available: True/False
 Models found: VGG16, EfficientNet-B0
 Installation successful!
\\\

---

##  Quick Start

### 1. Predict from Single Spectrogram

\\\python
from src.predictor import EarthquakePrecursorPredictor

# Initialize predictor with EfficientNet-B0 (recommended)
predictor = EarthquakePrecursorPredictor(
    model_name='efficientnet',
    model_path='models/best_efficientnet_smote_model.pth'
)

# Predict from spectrogram image
result = predictor.predict('data/sample_spectrogram.png')

print(f"Magnitude: {result['magnitude']} (confidence: {result['mag_confidence']:.2%})")
print(f"Azimuth: {result['azimuth']} (confidence: {result['azi_confidence']:.2%})")
print(f"Is Precursor: {result['is_precursor']}")
\\\

### 2. Batch Prediction

\\\python
from src.predictor import EarthquakePrecursorPredictor

predictor = EarthquakePrecursorPredictor(model_name='efficientnet')

# Predict multiple spectrograms
image_paths = [
    'data/sample1.png',
    'data/sample2.png',
    'data/sample3.png'
]

results = predictor.predict_batch(image_paths)

for img_path, result in zip(image_paths, results):
    print(f"{img_path}: {result['magnitude']} - {result['azimuth']}")
\\\

### 3. Generate Grad-CAM Visualization

\\\python
from src.explainability import generate_gradcam

# Generate Grad-CAM for interpretability
gradcam_result = generate_gradcam(
    model_path='models/best_efficientnet_smote_model.pth',
    image_path='data/sample_spectrogram.png',
    output_path='output/gradcam_visualization.png'
)

print(f"Grad-CAM saved to: {gradcam_result['output_path']}")
print(f"Prediction: {gradcam_result['prediction']}")
print(f"Attention focus: {gradcam_result['attention_summary']}")
\\\

---

##  Usage Examples

### Example 1: Train VGG16 Model

\\\python
from src.train import train_model

# Train VGG16 with default hyperparameters
train_model(
    model_name='vgg16',
    data_dir='dataset_spectrogram',
    output_dir='experiments/vgg16_training',
    epochs=50,
    batch_size=32,
    learning_rate=1e-4,
    early_stopping_patience=10
)
\\\

### Example 2: Train EfficientNet with Hyperparameter Tuning

\\\python
from src.train import train_with_optuna

# Automated hyperparameter tuning with Optuna
best_params = train_with_optuna(
    model_name='efficientnet',
    data_dir='dataset_spectrogram',
    output_dir='experiments/efficientnet_tuning',
    n_trials=20,
    timeout=3600  # 1 hour
)

print(f"Best hyperparameters: {best_params}")
\\\

### Example 3: Evaluate Model

\\\python
from src.evaluate import evaluate_model

# Comprehensive evaluation
results = evaluate_model(
    model_path='models/best_efficientnet_smote_model.pth',
    test_data_dir='dataset_spectrogram/test',
    output_dir='results/evaluation'
)

print(f"Magnitude Accuracy: {results['magnitude_accuracy']:.2%}")
print(f"Azimuth Accuracy: {results['azimuth_accuracy']:.2%}")
print(f"Confusion matrices saved to: {results['output_dir']}")
\\\

### Example 4: Compare Models

\\\python
from src.compare import compare_models

# Compare VGG16 vs EfficientNet
comparison = compare_models(
    model1_path='models/best_vgg16_model.pth',
    model2_path='models/best_efficientnet_smote_model.pth',
    test_data_dir='dataset_spectrogram/test',
    output_dir='results/comparison'
)

print(f"Comparison report saved to: {comparison['report_path']}")
\\\

### Example 5: LOEO Cross-Validation

\\\python
from src.validation import loeo_cross_validation

# Leave-One-Event-Out validation
loeo_results = loeo_cross_validation(
    model_name='efficientnet',
    data_dir='dataset_spectrogram',
    n_folds=10,
    output_dir='results/loeo_validation'
)

print(f"LOEO Magnitude Accuracy: {loeo_results['magnitude_acc_mean']:.2%}  {loeo_results['magnitude_acc_std']:.2%}")
print(f"LOEO Azimuth Accuracy: {loeo_results['azimuth_acc_mean']:.2%}  {loeo_results['azimuth_acc_std']:.2%}")
\\\

### Example 6: Real-time Monitoring

\\\python
from src.monitor import RealtimeMonitor

# Real-time precursor monitoring
monitor = RealtimeMonitor(
    model_path='models/best_efficientnet_smote_model.pth',
    data_source='bmkg_api',  # or 'local_files'
    alert_threshold=0.7
)

# Start monitoring
monitor.start()

# Monitor will automatically:
# 1. Fetch geomagnetic data every hour
# 2. Generate spectrograms
# 3. Run predictions
# 4. Send alerts if precursor detected
\\\

---

##  Model Comparison

### Architecture Comparison

| Feature | VGG16 | EfficientNet-B0 |
|---------|-------|-----------------|
| **Year** | 2014 | 2019 |
| **Parameters** | 245M | 4.7M |
| **Model Size** | 528 MB | 20 MB |
| **Layers** | 16 (13 conv + 3 FC) | 8 MBConv blocks |
| **Convolution** | Standard | Depthwise Separable |
| **Attention** | None | SE blocks |
| **Activation** | ReLU | SiLU (Swish) |
| **Skip Connections** | No | Yes |

### Performance Comparison

| Metric | VGG16 | EfficientNet-B0 | Winner |
|--------|-------|-----------------|--------|
| **Magnitude Accuracy** | 98.68% | 94.37% | VGG16 |
| **Azimuth Accuracy** | 54.93% | 57.39% | EfficientNet |
| **Inference Speed** | 125 ms | 50 ms | EfficientNet (2.5) |
| **Training Time** | 2.3 hours | 3.8 hours | VGG16 |
| **Generalization** | Moderate | Excellent | EfficientNet |

### When to Use Each Model

**Use VGG16 when**:
- Maximum magnitude accuracy is critical
- Computational resources are abundant
- Training time is limited
- Research/analysis purposes

**Use EfficientNet-B0 when**:
- Production deployment required
- Mobile/edge deployment needed
- Real-time inference required
- Better azimuth accuracy desired
- Resource constraints exist

---

##  Dataset

### Data Source

- **Geomagnetic Data**: BMKG (Indonesian Meteorological, Climatological, and Geophysical Agency)
- **Earthquake Catalog**: BMKG Earthquake Catalog
- **Period**: 2018-2025
- **Stations**: 25 geomagnetic stations across Indonesia
- **Components**: 3-component magnetometer (H, D, Z)

### Dataset Statistics

- **Unique Earthquake Events**: 256
- **Total Samples**: 1,972 spectrograms
  - Precursor samples: 1,084 (from 256 events)
  - Normal samples: 888 (from ~200 quiet days)
- **Temporal Windowing**: 4.2 multiplication factor
- **Magnitude Range**: M4.0 - M7.0+
- **Azimuth Coverage**: All 8 directions + Normal

### Data Split

- **Training**: 1,336 samples (67.7%)
- **Validation**: 352 samples (17.9%)
- **Test**: 284 samples (14.4%)
- **Split Method**: Fixed by station+date (no data leakage)

### Download Dataset

Due to size constraints, the full dataset is hosted externally:

\\\ash
# Download dataset (requires authentication)
python scripts/download_dataset.py --output data/

# Or manually download from:
# https://drive.google.com/... (link will be provided upon publication)
\\\

---

##  Project Structure

\\\
earthquake-precursor-cnn/
 README.md                          # This file
 LICENSE                            # MIT License
 requirements.txt                   # Python dependencies
 setup.py                           # Package setup
 .gitignore                         # Git ignore rules

 src/                               # Source code
    __init__.py
    models/                        # Model architectures
       vgg16_multitask.py
       efficientnet_multitask.py
       base_model.py
    data/                          # Data processing
       dataset.py
       preprocessing.py
       augmentation.py
    train.py                       # Training scripts
    evaluate.py                    # Evaluation scripts
    predictor.py                   # Inference API
    explainability.py              # Grad-CAM implementation
    validation.py                  # LOEO validation
    compare.py                     # Model comparison
    utils.py                       # Utility functions

 models/                            # Pre-trained models
    best_vgg16_model.pth
    best_efficientnet_smote_model.pth
    README.md

 data/                              # Dataset (not included in repo)
    README.md                      # Dataset documentation
    sample_spectrograms/           # Sample data for testing
    download_instructions.md

 notebooks/                         # Jupyter notebooks
    01_data_exploration.ipynb
    02_model_training.ipynb
    03_model_evaluation.ipynb
    04_gradcam_visualization.ipynb
    05_model_comparison.ipynb

 scripts/                           # Utility scripts
    download_models.py
    download_dataset.py
    verify_installation.py
    generate_spectrograms.py
    run_experiments.sh

 tests/                             # Unit tests
    test_models.py
    test_data.py
    test_training.py
    test_inference.py

 docs/                              # Documentation
    paper.pdf                      # Research paper
    supplementary.pdf              # Supplementary materials
    API.md                         # API documentation
    TRAINING.md                    # Training guide
    DEPLOYMENT.md                  # Deployment guide

 figures/                           # Figures for paper
    architecture_vgg16.png
    architecture_efficientnet.png
    gradcam_comparison.png
    confusion_matrices.png
    training_curves.png

 results/                           # Experimental results
     vgg16/
        training_history.csv
        test_results.json
        confusion_matrix.png
     efficientnet/
        training_history.csv
        test_results.json
        confusion_matrix.png
     comparison/
         model_comparison.csv
         comparison_report.md
\\\

---

##  Citation

If you use this code or models in your research, please cite our paper:

\\\ibtex
@article{earthquake_precursor_2026,
  title={Earthquake Precursor Detection using Deep Learning: A Comparative Study of VGG16 and EfficientNet-B0},
  author={[Your Name] and [Co-authors]},
  journal={[Journal Name]},
  year={2026},
  volume={[Volume]},
  pages={[Pages]},
  doi={[DOI]}
}
\\\

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

##  Acknowledgments

- **BMKG** for providing geomagnetic and earthquake data
- **PyTorch** team for the deep learning framework
- **Optuna** team for hyperparameter optimization library
- **EfficientNet** authors (Tan & Le, 2019) for the architecture
- **VGG** authors (Simonyan & Zisserman, 2014) for the architecture
- **Grad-CAM** authors (Selvaraju et al., 2017) for explainability method

---

##  Contact

For questions, issues, or collaborations:

- **Email**: [your.email@institution.edu]
- **GitHub Issues**: [https://github.com/yourusername/earthquake-precursor-cnn/issues](https://github.com/yourusername/earthquake-precursor-cnn/issues)
- **Research Gate**: [Your Profile]
- **Google Scholar**: [Your Profile]

---

##  Related Resources

- **Paper**: [Link to published paper]
- **Supplementary Materials**: [docs/supplementary.pdf](docs/supplementary.pdf)
- **Dataset**: [Link to dataset]
- **Pre-trained Models**: [GitHub Releases](https://github.com/yourusername/earthquake-precursor-cnn/releases)
- **Demo**: [Link to online demo]

---

##  Project Status

-  Paper submitted to [Journal Name]
-  Code released
-  Pre-trained models available
-  Dataset release pending approval
-  Online demo in development

---

**Last Updated**: February 4, 2026  
**Version**: 1.0.0  
**Status**: Active Development

