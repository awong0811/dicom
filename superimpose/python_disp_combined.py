import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="display",
        description="Display a superimposed image of US and MRI data"
    )
    parser.add_argument(
        '-s', '--save',
        type=str,
        help='Save the image to a destination on your computer'
    )
    args = parser.parse_args()

    save = args.save

    mat_contents = scipy.io.loadmat('./matlab_files/20252001_US_image_data.mat')
    raw_img = mat_contents['mid_sag_img_data']
    img_data = raw_img.copy()

    # Apply smoothing
    args_bf = {'type': 'bilateral_filter', 'd': 9, 'sigmaColor': 75, 'sigmaSpace': 75}
    args_nlm = {'type': 'nl_means', 'h': 10, 'templateWindowSize': 7, 'searchWindowSize': 21}
    smoothed = smooth(img_data, args_bf)

    threshold = 165
    thresholded = np.where(smoothed > threshold, smoothed, 0)
    cropped = crop(thresholded)
    # Display the image
    newscale = 0.04933945
    us_H, us_W = (cropped.shape[0]-1)*newscale, (cropped.shape[1]-1)*newscale
    plt.imshow(cropped, extent=(0, us_H, 0, us_W), aspect='auto')
    plt.colorbar()
    plt.title('US Mid-Slice Image')
    plt.gca().invert_yaxis()  # Match MATLAB's Y-axis direction
    plt.show()

    # Load and display the MRI image
    mri_mat_contents = scipy.io.loadmat(r'.\matlab_files\20252001_MRI_image_data.mat')
    mri = mri_mat_contents['mid_slice_img_data']
    newMRIC = mri_mat_contents['newMRIC'].flatten()  # Make sure you're using the right variable
    newscale_mri = mri_mat_contents['newscale'].flatten()  # Use a different variable to avoid confusion
    mri = np.flip(mri, axis=1)
    mri *= 255/np.max(mri)
    alpha = 2.0  # Contrast control (1.0 = original, >1.0 = higher contrast)
    beta = 0  # Brightness control (optional)
    mri = sharpen(mri)
    mri = np.clip(alpha * mri + beta, 0, 255)
    # plt.imsave(r"C:\Users\awong\Downloads\mri_scan_head.png", np.flip(mri, axis=0), cmap='gray')
    # Display the MRI image
    mri_H, mri_W = (newMRIC[0]-1)*newscale_mri[0], (newMRIC[1]-1)*newscale_mri[1]
    plt.imshow(mri, extent=(0, mri_H, 0, mri_W), aspect='auto')
    plt.colorbar()
    plt.title('MRI Mid-Slice Image')
    plt.gca().invert_yaxis()  # Match MATLAB's Y-axis direction
    plt.show()

    # row = 40
    # col = 205
    # scale = 1.1
    # angle = 45

    row = 60
    col = 255
    scale = 1
    angle = 10

    # row = 33
    # col = 205
    # scale = 1.17
    # angle = 45

    # Rotate ultrasound scan
    cropped = np.flip(cropped, axis=1)
    rotated = rotate_image(cropped, angle=angle)
    scaled = cv2.resize(rotated, dsize=None, fx=scale, fy=scale)
    H, W = scaled.shape

    mri[row:row+H, col:col+W] += scaled
    mri = np.clip(mri, 0, 255)
    # Display the superimposed image
    mri_H, mri_W = (newMRIC[0]-1)*newscale_mri[0], (newMRIC[1]-1)*newscale_mri[1]
    print(f'Height of image {mri_H}')
    print(f'Width of image {mri_W}')
    plt.imshow(mri, extent=(0, mri_H, 0, mri_W), aspect='auto')
    plt.colorbar()
    plt.title('Superimposed Image')
    plt.gca().invert_yaxis()  # Match MATLAB's Y-axis direction
    plt.show()
    if save:
        plt.imsave(save, np.flip(mri, axis=0), cmap='gray')