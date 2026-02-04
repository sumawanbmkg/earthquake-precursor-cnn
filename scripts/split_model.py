#!/usr/bin/env python3
"""
Split large model files into smaller parts for GitHub Releases upload.
Each part will be < 25MB to comply with browser upload limits.

Usage:
    python scripts/split_model.py --input models/best_vgg16_model.pth --output models/parts/
"""

import os
import sys
import argparse
import hashlib
from pathlib import Path

CHUNK_SIZE = 24 * 1024 * 1024  # 24MB per part (under 25MB limit)


def split_file(input_path: Path, output_dir: Path) -> dict:
    """Split a file into smaller parts."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_size = input_path.stat().st_size
    num_parts = (file_size + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    print(f"Splitting {input_path.name}")
    print(f"  Size: {file_size / (1024*1024):.2f} MB")
    print(f"  Parts: {num_parts}")
    
    # Calculate original file hash
    print("  Calculating checksum...")
    md5_hash = hashlib.md5()
    with open(input_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5_hash.update(chunk)
    original_md5 = md5_hash.hexdigest()
    
    # Split file
    parts = []
    with open(input_path, 'rb') as f:
        for i in range(num_parts):
            part_name = f"{input_path.stem}.part{i+1:03d}"
            part_path = output_dir / part_name
            
            chunk = f.read(CHUNK_SIZE)
            with open(part_path, 'wb') as part_file:
                part_file.write(chunk)
            
            part_size = len(chunk)
            parts.append({
                'name': part_name,
                'size': part_size,
                'path': str(part_path)
            })
            
            print(f"  Created: {part_name} ({part_size / (1024*1024):.2f} MB)")
    
    # Create manifest file
    manifest = {
        'original_file': input_path.name,
        'original_size': file_size,
        'original_md5': original_md5,
        'num_parts': num_parts,
        'chunk_size': CHUNK_SIZE,
        'parts': [p['name'] for p in parts]
    }
    
    manifest_path = output_dir / f"{input_path.stem}.manifest.txt"
    with open(manifest_path, 'w') as f:
        f.write(f"# Manifest for {input_path.name}\n")
        f.write(f"original_file={input_path.name}\n")
        f.write(f"original_size={file_size}\n")
        f.write(f"original_md5={original_md5}\n")
        f.write(f"num_parts={num_parts}\n")
        for part in parts:
            f.write(f"part={part['name']}\n")
    
    print(f"  Manifest: {manifest_path.name}")
    print(f"  MD5: {original_md5}")
    
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Split large model files')
    parser.add_argument('--input', '-i', required=True, help='Input model file')
    parser.add_argument('--output', '-o', default='models/parts/', help='Output directory')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    manifest = split_file(input_path, Path(args.output))
    
    print("\n" + "=" * 50)
    print("Split complete!")
    print(f"Upload all .part* files and .manifest.txt to GitHub Releases")
    print("Users can reassemble with: python scripts/merge_model.py")


if __name__ == '__main__':
    main()
