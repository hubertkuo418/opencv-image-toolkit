import cv2

def equalize(gray_img):
    """Histogram Equalization for contrast enhancement"""
    return cv2.equalizeHist(gray_img)