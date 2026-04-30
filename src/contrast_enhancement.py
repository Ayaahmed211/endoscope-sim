import cv2

def enhance_contrast(image):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Works on the L channel of LAB color space so color is not distorted.
    Fixes uneven endoscopic lighting (bright center, dark edges).
    
    clipLimit=2.0     → prevents over-amplifying noise
    tileGridSize=(8,8) → divides image into local regions for equalization
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)