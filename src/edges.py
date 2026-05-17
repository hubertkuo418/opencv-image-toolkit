import cv2

def canny_edge(img, low=100, high=200):
    """Canny edge detection"""
    return cv2.Canny(img, low, high)