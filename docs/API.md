# API Documentation

## Predictor API

### EarthquakePrecursorPredictor

Main class for earthquake precursor prediction.

#### Initialization

\\\python
from src.predictor import EarthquakePrecursorPredictor

predictor = EarthquakePrecursorPredictor(
    model_name='efficientnet',  # or 'vgg16'
    model_path='models/best_efficientnet_smote_model.pth',
    device='cpu'  # or 'cuda'
)
\\\

**Parameters**:
- \model_name\ (str): Model architecture ('vgg16' or 'efficientnet')
- \model_path\ (str): Path to pre-trained model weights
- \device\ (str, optional): Device for inference ('cpu' or 'cuda'). Default: 'cpu'

#### Methods

##### predict()

Predict from a single spectrogram image.

\\\python
result = predictor.predict(
    image_path='data/sample.png',
    return_probabilities=True
)
\\\

**Parameters**:
- \image_path\ (str): Path to spectrogram image
- \eturn_probabilities\ (bool, optional): Return class probabilities. Default: True

**Returns** (dict):
\\\python
{
    'magnitude': 'Medium',  # Predicted magnitude class
    'azimuth': 'SE',  # Predicted azimuth direction
    'is_precursor': True,  # Whether precursor detected
    'mag_confidence': 0.95,  # Magnitude prediction confidence
    'azi_confidence': 0.78,  # Azimuth prediction confidence
    'mag_probabilities': [0.01, 0.02, 0.95, 0.02],  # All class probabilities
    'azi_probabilities': [0.01, 0.05, 0.08, 0.12, 0.78, 0.03, 0.02, 0.01, 0.01]
}
\\\

##### predict_batch()

Predict from multiple spectrogram images.

\\\python
results = predictor.predict_batch(
    image_paths=['data/sample1.png', 'data/sample2.png'],
    batch_size=32
)
\\\

**Parameters**:
- \image_paths\ (list): List of paths to spectrogram images
- \atch_size\ (int, optional): Batch size for inference. Default: 32

**Returns** (list): List of prediction dictionaries (same format as \predict()\)

---

## Training API

### train_model()

Train a model with specified hyperparameters.

\\\python
from src.train import train_model

results = train_model(
    model_name='efficientnet',
    data_dir='dataset_spectrogram',
    output_dir='experiments/my_training',
    epochs=50,
    batch_size=32,
    learning_rate=1e-4,
    weight_decay=1e-4,
    dropout=0.5,
    early_stopping_patience=10,
    device='cpu'
)
\\\

**Parameters**:
- \model_name\ (str): Model architecture ('vgg16' or 'efficientnet')
- \data_dir\ (str): Path to dataset directory
- \output_dir\ (str): Path to save training outputs
- \epochs\ (int): Maximum number of training epochs
- \atch_size\ (int): Training batch size
- \learning_rate\ (float): Initial learning rate
- \weight_decay\ (float): L2 regularization weight
- \dropout\ (float): Dropout rate
- \early_stopping_patience\ (int): Early stopping patience
- \device\ (str): Device for training ('cpu' or 'cuda')

**Returns** (dict):
\\\python
{
    'best_epoch': 11,
    'best_val_loss': 0.8523,
    'best_mag_acc': 0.9773,
    'best_azi_acc': 0.7727,
    'model_path': 'experiments/my_training/best_model.pth',
    'history_path': 'experiments/my_training/training_history.csv'
}
\\\

### train_with_optuna()

Train with automated hyperparameter tuning using Optuna.

\\\python
from src.train import train_with_optuna

best_params = train_with_optuna(
    model_name='efficientnet',
    data_dir='dataset_spectrogram',
    output_dir='experiments/tuning',
    n_trials=20,
    timeout=3600
)
\\\

**Parameters**:
- \model_name\ (str): Model architecture
- \data_dir\ (str): Path to dataset
- \output_dir\ (str): Path to save outputs
- \
_trials\ (int): Number of Optuna trials
- \	imeout\ (int, optional): Timeout in seconds

**Returns** (dict): Best hyperparameters found

---

## Evaluation API

### evaluate_model()

Comprehensive model evaluation.

\\\python
from src.evaluate import evaluate_model

results = evaluate_model(
    model_path='models/best_efficientnet_smote_model.pth',
    test_data_dir='dataset_spectrogram/test',
    output_dir='results/evaluation',
    generate_visualizations=True
)
\\\

**Parameters**:
- \model_path\ (str): Path to trained model
- \	est_data_dir\ (str): Path to test data
- \output_dir\ (str): Path to save results
- \generate_visualizations\ (bool): Generate plots

**Returns** (dict):
\\\python
{
    'magnitude_accuracy': 0.9437,
    'azimuth_accuracy': 0.5739,
    'magnitude_precision': 0.9456,
    'magnitude_recall': 0.9437,
    'magnitude_f1': 0.9441,
    'azimuth_precision': 0.6123,
    'azimuth_recall': 0.5739,
    'azimuth_f1': 0.5845,
    'confusion_matrix_mag': [[...], [...]],
    'confusion_matrix_azi': [[...], [...]],
    'output_dir': 'results/evaluation'
}
\\\

---

## Explainability API

### generate_gradcam()

Generate Grad-CAM visualization.

\\\python
from src.explainability import generate_gradcam

result = generate_gradcam(
    model_path='models/best_efficientnet_smote_model.pth',
    image_path='data/sample.png',
    output_path='output/gradcam.png',
    target_class='magnitude'  # or 'azimuth'
)
\\\

**Parameters**:
- \model_path\ (str): Path to trained model
- \image_path\ (str): Path to input spectrogram
- \output_path\ (str): Path to save visualization
- \	arget_class\ (str): Target task ('magnitude' or 'azimuth')

**Returns** (dict):
\\\python
{
    'output_path': 'output/gradcam.png',
    'prediction': 'Medium',
    'confidence': 0.95,
    'attention_summary': 'Focus on ULF bands (0.001-0.01 Hz)'
}
\\\

---

## Validation API

### loeo_cross_validation()

Leave-One-Event-Out cross-validation.

\\\python
from src.validation import loeo_cross_validation

results = loeo_cross_validation(
    model_name='efficientnet',
    data_dir='dataset_spectrogram',
    n_folds=10,
    output_dir='results/loeo'
)
\\\

**Parameters**:
- \model_name\ (str): Model architecture
- \data_dir\ (str): Path to dataset
- \
_folds\ (int): Number of folds
- \output_dir\ (str): Path to save results

**Returns** (dict):
\\\python
{
    'magnitude_acc_mean': 0.9423,
    'magnitude_acc_std': 0.021,
    'azimuth_acc_mean': 0.5218,
    'azimuth_acc_std': 0.034,
    'fold_results': [...],
    'output_dir': 'results/loeo'
}
\\\

---

## Data API

### EarthquakeDataset

PyTorch Dataset for earthquake precursor data.

\\\python
from src.data.dataset import EarthquakeDataset

dataset = EarthquakeDataset(
    data_dir='dataset_spectrogram/train',
    transform=None,
    augment=True
)
\\\

**Parameters**:
- \data_dir\ (str): Path to data directory
- \	ransform\ (callable, optional): Transform to apply
- \ugment\ (bool): Apply data augmentation

**Methods**:
- \__len__()\: Returns dataset size
- \__getitem__(idx)\: Returns (image, magnitude_label, azimuth_label)

---

For more examples, see the [notebooks/](../notebooks/) directory.
