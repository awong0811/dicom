import cv2
import numpy as np

def smooth(img_data: np.ndarray, args: dict) -> np.ndarray:
    type = args["type"]
    args = {k: v for k, v in args.items() if k != 'type'}
    # args.pop('type', None)
    if type == 'bilateral_filter':
        img_data = img_data.astype(np.float32)
        # img_data = cv2.bilateralFilter(img_data, d=9, sigmaColor=75, sigmaSpace=75)
        img_data = cv2.bilateralFilter(img_data, **args)
        img_data = img_data.astype(np.float64)
        return img_data
    elif type == 'nl_means':
        img_data = img_data.astype(np.uint8)
        # img_data = cv2.fastNlMeansDenoising(img_data, None, h=10, templateWindowSize=7, searchWindowSize=21)
        img_data = cv2.fastNlMeansDenoising(img_data, None, **args)
        img_data = img_data.astype(np.float64)
        return img_data

