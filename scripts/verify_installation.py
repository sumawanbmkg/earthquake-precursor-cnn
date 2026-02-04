#!/usr/bin/env python3
"""
Verify installation of earthquake precursor detection system.

Usage:
    python scripts/verify_installation.py
"""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python version: {version.major}.{version.minor}.{version.micro} (requires 3.10+)")
        return False


def check_pytorch():
    """Check PyTorch installation."""
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("○ CUDA not available (CPU mode)")
        
        return True
    except ImportError:
        print("✗ PyTorch not installed")
        return False


def check_dependencies():
    """Check other dependencies."""
    dependencies = [
        ('numpy', 'numpy'),
        ('PIL', 'Pillow'),
        ('matplotlib', 'matplotlib'),
        ('sklearn', 'scikit-learn'),
        ('tqdm', 'tqdm'),
        ('requests', 'requests'),
    ]
    
    all_ok = True
    for module_name, package_name in dependencies:
        try:
            __import__(module_name)
            print(f"✓ {package_name} installed")
        except ImportError:
            print(f"✗ {package_name} not installed")
            all_ok = False
    
    return all_ok


def check_models():
    """Check if model files exist."""
    models_dir = Path('models')
    
    models = [
        ('best_efficientnet_smote_model.pth', 'EfficientNet-B0'),
        ('best_vgg16_model.pth', 'VGG16'),
    ]
    
    found_any = False
    for filename, name in models:
        filepath = models_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✓ {name} model found ({size_mb:.1f} MB)")
            found_any = True
        else:
            print(f"○ {name} model not found (run: python scripts/download_models.py)")
    
    return found_any


def check_source_files():
    """Check if source files exist."""
    src_dir = Path('src')
    
    required_files = [
        'predictor.py',
        'train.py',
        'evaluate.py',
        'explainability.py',
    ]
    
    all_ok = True
    for filename in required_files:
        filepath = src_dir / filename
        if filepath.exists():
            print(f"✓ {filename} found")
        else:
            print(f"✗ {filename} not found")
            all_ok = False
    
    return all_ok


def main():
    print("=" * 60)
    print("Earthquake Precursor CNN - Installation Verification")
    print("=" * 60)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("PyTorch", check_pytorch),
        ("Dependencies", check_dependencies),
        ("Source Files", check_source_files),
        ("Models", check_models),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n[{name}]")
        results.append(check_func())
    
    print("\n" + "=" * 60)
    
    if all(results):
        print("✓ Installation successful!")
        print("\nYou can now use the system:")
        print("  from src.predictor import EarthquakePrecursorPredictor")
        print("  predictor = EarthquakePrecursorPredictor(model_name='efficientnet')")
        print("  result = predictor.predict('path/to/spectrogram.png')")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        if not results[4]:  # Models check
            print("\nTo download models:")
            print("  python scripts/download_models.py")
        sys.exit(1)


if __name__ == '__main__':
    main()
