import cv2

def reduce_noise(image):
    """
    Non-Local Means Denoising.
    Removes camera/sensor noise while preserving edges like polyp boundaries.
    
    h=10        → filter strength (higher = more smoothing but may blur)
    hColor=10   → same but for color channels
    templateWindowSize=7  → patch size for weight computation
    searchWindowSize=21   → search area for similar patches
    """
    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h=10,
        hColor=10,
        templateWindowSize=7,
        searchWindowSize=21
    )