<<<<<<< HEAD
"""
=============================================================================
  ENDOSCOPIC IMAGE PROCESSING — BONUS TASK  (SBE3220 Task 02)
=============================================================================
  Run options:
    python main.py                                     → processes data/PNG/Original (all images)
    python main.py --limit 5                           → processes first 5 images only
    python main.py --image data/PNG/Original/img.png  → single image
    python main.py --folder data/TIF/Original          → whole folder
    python main.py --format PNG --limit 10             → first 10 PNG images
    python main.py --format TIF --limit 5              → first 5 TIF images
=============================================================================
"""

import cv2
import os
import argparse
import numpy as np
import pandas as pd

from src.noise_reduction      import reduce_noise
from src.contrast_enhancement import enhance_contrast
from src.feature_extraction   import (extract_shape_features,
                                      extract_color_features,
                                      extract_texture_features)
from src.visualization        import visualize_pipeline, visualize_features


# ─────────────────────────────────────────────
def process_image(image_path, visualize=True):
    """Full pipeline for one image."""

    print(f"\n{'='*55}")
    print(f"  Processing : {os.path.basename(image_path)}")
    print(f"{'='*55}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"  [ERROR] Cannot read: {image_path}")
        return None

    print(f"  Size       : {image.shape[1]} x {image.shape[0]} px")

    # ── Pipeline ───────────────────────────────
    denoised = reduce_noise(image)
    print("  [✓] Noise Reduction  (Non-Local Means)")

    enhanced = enhance_contrast(denoised)
    print("  [✓] Contrast Enhancement (CLAHE)")

    # ── Features ───────────────────────────────
    shape   = extract_shape_features(enhanced)
    color   = extract_color_features(enhanced)
    texture = extract_texture_features(enhanced)
    print("  [✓] Features Extracted (Shape + Color + Texture)")

    # ── Print to terminal ───────────────────────
    print("\n  >> SHAPE FEATURES:")
    for k, v in shape.items():
        print(f"      {k:<18} = {v}")

    print("\n  >> COLOR FEATURES:")
    for k, v in color.items():
        print(f"      {k:<18} = {v}")

    print("\n  >> TEXTURE FEATURES (GLCM):")
    for k, v in texture.items():
        print(f"      {k:<25} = {v}")

    # ── Build combined feature dict ─────────────
    name = os.path.splitext(os.path.basename(image_path))[0]
    all_features = {'image': os.path.basename(image_path)}
    all_features.update(shape)
    all_features.update(color)
    all_features.update(texture)

    # ── Visualize ──────────────────────────────
    if visualize:
        visualize_pipeline(image, denoised, enhanced, name)
        visualize_features(all_features, name)

    return all_features


# ─────────────────────────────────────────────
def process_folder(folder_path, limit=None, visualize_count=1):
    """Processes images in a folder and saves a CSV summary."""

    exts   = ('.png', '.PNG', '.tif', '.TIF', '.tiff', '.TIFF', '.jpg', '.jpeg')
    images = [f for f in os.listdir(folder_path) if f.endswith(exts)]

    if not images:
        print(f"[ERROR] No images found in: {folder_path}")
        return

    # ── Apply limit ────────────────────────────
    if limit:
        images = images[:limit]
        print(f"\nProcessing {limit} image(s) from: {folder_path}")
    else:
        print(f"\nFound {len(images)} image(s) in: {folder_path}")

    results = []

    for i, fname in enumerate(images):
        # visualize only the first `visualize_count` images
        should_visualize = i < visualize_count
        feats = process_image(
            os.path.join(folder_path, fname),
            visualize=should_visualize
        )
        if feats:
            results.append(feats)

    # ── Save CSV ───────────────────────────────
    if results:
        os.makedirs('output/results', exist_ok=True)
        df = pd.DataFrame(results)
        csv_path = 'output/results/features.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n[Saved] {csv_path}")
        print("\n" + df.to_string(index=False))


# ─────────────────────────────────────────────
def run_synthetic_test():
    """Generates a fake endoscopic image and runs the full pipeline on it."""

    print("\n[TEST MODE] No data folder found. Running synthetic test...\n")

    test = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(test, (200, 200), 180, (60, 120, 80), -1)
    cv2.circle(test, (220, 190),  50, (30,  50, 150), -1)
    noise = np.random.normal(0, 25, test.shape).astype(np.int16)
    test  = np.clip(test.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    os.makedirs('data', exist_ok=True)
    test_path = 'data/test_endoscopy.png'
    cv2.imwrite(test_path, test)
    print(f"[Saved] {test_path}")
    process_image(test_path)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Endoscopic Image Processing - Bonus Task")
    parser.add_argument('--image',   type=str, help='Path to a single image file')
    parser.add_argument('--folder',  type=str, help='Path to a folder of images')
    parser.add_argument('--format',  type=str, choices=['PNG', 'TIF'],
                        help='Process Original subfolder of PNG or TIF')
    parser.add_argument('--limit',   type=int, default=None,
                        help='Max number of images to process (default: all)')
    parser.add_argument('--visualize', type=int, default=1,
                        help='How many images to show plots for (default: 1)')
    args = parser.parse_args()

    if args.image:
        process_image(args.image, visualize=True)

    elif args.folder:
        process_folder(args.folder, limit=args.limit, visualize_count=args.visualize)

    elif args.format:
        target = os.path.join('data', args.format, 'Original')
        if os.path.exists(target):
            process_folder(target, limit=args.limit, visualize_count=args.visualize)
        else:
            print(f"[ERROR] Folder not found: {target}")

    else:
        default = os.path.join('data', 'PNG', 'Original')
        if os.path.exists(default):
            process_folder(default, limit=args.limit, visualize_count=args.visualize)
        else:
            fallback = os.path.join('data', 'TIF', 'Original')
            if os.path.exists(fallback):
                print(f"[INFO] PNG/Original not found — falling back to: {fallback}")
                process_folder(fallback, limit=args.limit, visualize_count=args.visualize)
            else:
                run_synthetic_test()
=======

import sys
from modules.app import EndoscopeApp


def main():
    app = EndoscopeApp()
    app.run()


if __name__ == "__main__":
    main()
>>>>>>> 49a5b5e992919bea94af456db377ac2581be2382
