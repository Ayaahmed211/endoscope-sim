import cv2
import matplotlib.pyplot as plt
import os


def visualize_pipeline(original, denoised, enhanced, image_name="image"):
    """
    Shows the 3-step processing pipeline side by side and saves the figure.
    Original → Noise Reduced → Contrast Enhanced
    """
    os.makedirs('output/figures', exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Endoscopic Processing Pipeline — {image_name}', fontsize=14, fontweight='bold')

    images = [original, denoised, enhanced]
    titles = [
        'Original',
        'Noise Reduction\n(Non-Local Means)',
        'Contrast Enhanced\n(CLAHE)'
    ]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    save_path = f'output/figures/{image_name}_pipeline.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  [Saved] {save_path}")


def visualize_features(features_dict, image_name="image"):
    """
    Bar chart showing all extracted feature values in one figure.
    Grouped by: Shape | Color | Texture
    """
    os.makedirs('output/figures', exist_ok=True)

    # Separate features into groups
    shape_keys   = ['area', 'perimeter', 'circularity']
    color_keys   = ['mean_R', 'mean_G', 'mean_B', 'std_R', 'std_G', 'std_B']
    texture_keys = ['glcm_contrast', 'glcm_dissimilarity', 'glcm_homogeneity',
                    'glcm_energy', 'glcm_correlation']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Extracted Features — {image_name}', fontsize=14, fontweight='bold')

    groups = [
        (shape_keys,   'Shape Features',   'steelblue'),
        (color_keys,   'Color Features',   'mediumseagreen'),
        (texture_keys, 'Texture Features', 'tomato'),
    ]

    for ax, (keys, title, color) in zip(axes, groups):
        vals   = [features_dict.get(k, 0) for k in keys]
        labels = [k.replace('glcm_', '') for k in keys]
        ax.bar(labels, vals, color=color, edgecolor='black', linewidth=0.7)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=30)
        for i, v in enumerate(vals):
            ax.text(i, v * 1.01, f'{v:.3f}', ha='center', fontsize=8)

    plt.tight_layout()
    save_path = f'output/figures/{image_name}_features.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  [Saved] {save_path}")