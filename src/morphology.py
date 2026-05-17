import cv2
import numpy as np

def dilate(img, ksize=5):
    """Expand white regions"""
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)


def erode(img, ksize=5):
    """Shrink white regions"""
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.erode(img, kernel, iterations=1)