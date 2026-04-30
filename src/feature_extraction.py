import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


# ─────────────────────────────────────────────
#  SHAPE FEATURES
# ─────────────────────────────────────────────
def extract_shape_features(image):
    """
    Extracts shape features from the largest detected contour.

    - Area        : Total pixel area of the main region
    - Perimeter   : Boundary length of the contour
    - Circularity : 4π × Area / Perimeter²
                    1.0 = perfect circle, <1.0 = irregular shape
                    Useful to flag irregular lesions/polyps
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Otsu: automatically finds best threshold to separate tissue from background
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {'area': 0, 'perimeter': 0, 'circularity': 0}

    largest   = max(contours, key=cv2.contourArea)
    area      = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, closed=True)
    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0

    return {
        'area':        round(area, 2),
        'perimeter':   round(perimeter, 2),
        'circularity': round(circularity, 4)
    }


# ─────────────────────────────────────────────
#  COLOR FEATURES
# ─────────────────────────────────────────────
def extract_color_features(image):
    """
    Extracts mean and standard deviation for each RGB channel.

    - Mean : Dominant color tone per channel
    - Std  : Color variation/spread per channel

    6 values total → compact color signature.
    Example: inflamed tissue = higher R mean, bile = higher G mean.
    """
    b, g, r = cv2.split(image)  # OpenCV loads BGR

    features = {}
    for channel, name in zip([r, g, b], ['R', 'G', 'B']):
        features[f'mean_{name}'] = round(float(np.mean(channel)), 4)
        features[f'std_{name}']  = round(float(np.std(channel)),  4)

    return features


# ─────────────────────────────────────────────
#  TEXTURE FEATURES (GLCM)
# ─────────────────────────────────────────────
def extract_texture_features(image):
    """
    Extracts texture using GLCM (Gray-Level Co-occurrence Matrix).

    GLCM counts how often pairs of pixel values appear side by side.
    From this matrix we derive:

    - Contrast     : Intensity difference between neighbors (high = sharp edges)
    - Dissimilarity: Like contrast but less sensitive to extreme values
    - Homogeneity  : Closeness to diagonal (high = smooth/uniform texture)
    - Energy       : Uniformity of the GLCM (high = repetitive pattern)
    - Correlation  : Linear dependency of gray levels (high = strong structure)

    Image resized to 128×128 first for speed.
    Computed at 4 angles (0°, 45°, 90°, 135°) then averaged.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.resize(gray, (128, 128))

    glcm = graycomatrix(
        gray_small,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )

    return {
        'glcm_contrast':      round(float(graycoprops(glcm, 'contrast').mean()),      4),
        'glcm_dissimilarity': round(float(graycoprops(glcm, 'dissimilarity').mean()), 4),
        'glcm_homogeneity':   round(float(graycoprops(glcm, 'homogeneity').mean()),   4),
        'glcm_energy':        round(float(graycoprops(glcm, 'energy').mean()),         4),
        'glcm_correlation':   round(float(graycoprops(glcm, 'correlation').mean()),   4),
    }