#!/usr/bin/env python3
"""
Download pre-trained models from GitHub Releases.

Usage:
    python scripts/download_models.py [--model MODEL] [--output-dir DIR]

Options:
    --model MODEL       Model to download: 'efficientnet', 'vgg16', or 'all' (default: 'all')
    --output-dir DIR    Output directory (default: 'models/')
"""

import os
import sys
import argparse
import hashlib
from pathlib import Path

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Installing required packages...")
    os.system(f"{sys.executable} -m pip install requests tqdm")
    import requests
    from tqdm import tqdm


# Model configurations
MODELS = {
    'efficientnet': {
        'filename': 'best_efficientnet_smote_model.pth',
        'url': 'https://github.com/sumawanbmkg/earthquake-precursor-cnn/releases/download/v1.0.0/best_efficientnet_smote_model.pth',
        'size_mb': 54,
        'md5': None,  # Will be updated after upload
        'description': 'EfficientNet-B0 (recommended for production)'
    },
    'vgg16': {
        'filename': 'best_vgg16_model.pth',
        'url': 'https://github.com/sumawanbmkg/earthquake-precursor-cnn/releases/download/v1.0.0/best_vgg16_model.pth',
        'size_mb': 1256,
        'md5': None,  # Will be updated after upload
        'description': 'VGG16 (highest magnitude accuracy)'
    }
}


def download_file(url: str, output_path: Path, description: str = None) -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True, allow_redirects=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        desc = description or output_path.name
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        return True
    
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False


def verify_md5(filepath: Path, expected_md5: str) -> bool:
    """Verify file MD5 checksum."""
    if expected_md5 is None:
        return True
    
    md5_hash = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5_hash.update(chunk)
    
    return md5_hash.hexdigest() == expected_md5


def download_model(model_name: str, output_dir: Path) -> bool:
    """Download a specific model."""
    if model_name not in MODELS:
        print(f"Unknown model: {model_name}")
        print(f"Available models: {', '.join(MODELS.keys())}")
        return False
    
    model_info = MODELS[model_name]
    output_path = output_dir / model_info['filename']
    
    # Check if already exists
    if output_path.exists():
        print(f"Model already exists: {output_path}")
        response = input("Download again? [y/N]: ").strip().lower()
        if response != 'y':
            return True
    
    print(f"\nDownloading {model_info['description']}...")
    print(f"Size: ~{model_info['size_mb']} MB")
    print(f"URL: {model_info['url']}")
    
    success = download_file(
        url=model_info['url'],
        output_path=output_path,
        description=model_info['filename']
    )
    
    if success:
        # Verify checksum if available
        if model_info['md5']:
            print("Verifying checksum...")
            if verify_md5(output_path, model_info['md5']):
                print("✓ Checksum verified")
            else:
                print("✗ Checksum mismatch! File may be corrupted.")
                return False
        
        print(f"✓ Downloaded: {output_path}")
        return True
    
    return False


def main():
    parser = argparse.ArgumentParser(
        description='Download pre-trained earthquake precursor detection models'
    )
    parser.add_argument(
        '--model', '-m',
        choices=['efficientnet', 'vgg16', 'all'],
        default='all',
        help='Model to download (default: all)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('models'),
        help='Output directory (default: models/)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Earthquake Precursor CNN - Model Downloader")
    print("=" * 60)
    
    # Determine which models to download
    if args.model == 'all':
        models_to_download = list(MODELS.keys())
    else:
        models_to_download = [args.model]
    
    # Download models
    success_count = 0
    for model_name in models_to_download:
        if download_model(model_name, args.output_dir):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"Downloaded {success_count}/{len(models_to_download)} models")
    
    if success_count == len(models_to_download):
        print("✓ All models downloaded successfully!")
        print("\nNext steps:")
        print("  1. Verify installation: python scripts/verify_installation.py")
        print("  2. Run prediction: python -c \"from src.predictor import EarthquakePrecursorPredictor; ...\"")
    else:
        print("✗ Some downloads failed. Please try again.")
        sys.exit(1)


if __name__ == '__main__':
    main()
