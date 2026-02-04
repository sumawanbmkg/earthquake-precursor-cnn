#!/usr/bin/env python3
"""
Merge split model parts back into original file.

Usage:
    python scripts/merge_model.py --manifest models/parts/best_vgg16_model.manifest.txt --output models/
"""

import os
import sys
import argparse
import hashlib
from pathlib import Path


def merge_files(manifest_path: Path, output_dir: Path) -> Path:
    """Merge split parts back into original file."""
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    parts_dir = manifest_path.parent
    
    # Parse manifest
    manifest = {}
    parts = []
    
    with open(manifest_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            key, value = line.split('=', 1)
            if key == 'part':
                parts.append(value)
            else:
                manifest[key] = value
    
    original_file = manifest['original_file']
    original_size = int(manifest['original_size'])
    original_md5 = manifest['original_md5']
    
    print(f"Merging {len(parts)} parts into {original_file}")
    print(f"  Expected size: {original_size / (1024*1024):.2f} MB")
    print(f"  Expected MD5: {original_md5}")
    
    # Merge parts
    output_path = output_dir / original_file
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as outfile:
        for i, part_name in enumerate(parts):
            part_path = parts_dir / part_name
            if not part_path.exists():
                print(f"  Error: Part not found: {part_path}")
                sys.exit(1)
            
            with open(part_path, 'rb') as part_file:
                outfile.write(part_file.read())
            
            print(f"  Merged: {part_name} ({i+1}/{len(parts)})")
    
    # Verify
    print("  Verifying checksum...")
    md5_hash = hashlib.md5()
    with open(output_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5_hash.update(chunk)
    
    actual_md5 = md5_hash.hexdigest()
    actual_size = output_path.stat().st_size
    
    if actual_md5 == original_md5 and actual_size == original_size:
        print(f"\n✓ Verification successful!")
        print(f"  Output: {output_path}")
        print(f"  Size: {actual_size / (1024*1024):.2f} MB")
        print(f"  MD5: {actual_md5}")
        return output_path
    else:
        print(f"\n✗ Verification failed!")
        print(f"  Expected MD5: {original_md5}, Got: {actual_md5}")
        print(f"  Expected size: {original_size}, Got: {actual_size}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Merge split model parts')
    parser.add_argument('--manifest', '-m', required=True, help='Manifest file path')
    parser.add_argument('--output', '-o', default='models/', help='Output directory')
    
    args = parser.parse_args()
    
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: Manifest not found: {manifest_path}")
        sys.exit(1)
    
    merge_files(manifest_path, Path(args.output))


if __name__ == '__main__':
    main()
