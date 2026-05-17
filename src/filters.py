import cv2

def to_gray(img):
    """Convert BGR image to grayscale"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gaussian_blur(img, ksize=(9, 9)):
    """Reduce noise using Gaussian Blur"""
    return cv2.GaussianBlur(img, ksize, 0)