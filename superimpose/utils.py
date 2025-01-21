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

def rotate_image(image, angle):
    # Get the dimensions of the image
    height, width = image.shape
    
    # Calculate the center of the image
    center = (width / 2, height / 2)
    
    # Compute the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, scale=1.0)
    
    # Expand the rotation matrix to 3x3 to use with warpPerspective
    rotation_matrix_h = np.vstack([rotation_matrix, [0, 0, 1]])
    
    # Calculate the bounding box of the rotated image
    cos_theta = abs(rotation_matrix[0, 0])
    sin_theta = abs(rotation_matrix[0, 1])
    new_width = int((height * sin_theta) + (width * cos_theta))
    new_height = int((height * cos_theta) + (width * sin_theta))
    
    # Adjust the rotation matrix to include the translation for new center
    rotation_matrix_h[0, 2] += (new_width / 2) - center[0]
    rotation_matrix_h[1, 2] += (new_height / 2) - center[1]
    
    # Apply the warp perspective
    rotated_image = cv2.warpPerspective(image, rotation_matrix_h, (new_width, new_height))
    
    return rotated_image

def sharpen(img: np.ndarray):
    kernel = np.ones((3,3))*-1
    kernel[1,1] = 9
    padded_img = np.pad(img, pad_width=1)
    output = np.zeros_like(img)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i,j] = np.sum(padded_img[i:i+3,j:j+3] * kernel)
    return output

def crop(img: np.ndarray):
    for i in range(img.shape[0]):
        if not np.all(img[i]==0):
            img = img[i:]
            break
    for j in range(img.shape[0]-1, -1, -1):
        if not np.all(img[j]==0):
            img = img[:j+1]
            break
    for k in range(img.shape[1]):
        if not np.all(img[:,k]==0):
            img = img[:,k:]
            break
    for l in range(img.shape[1]-1, -1, -1):
        if not np.all(img[:,l]==0):
            img = img[:,:l+1]
            break
    return img