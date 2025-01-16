import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils import *


if __name__ == '__main__':
    mat_contents = scipy.io.loadmat('./matlab_files/mid_saggital_image_data.mat')
    raw_img = mat_contents['mid_sag_img_data']
    print ("Image Data: ", raw_img.shape)
    img_data = raw_img.copy()

    # Apply smoothing
    args_bf = {'type': 'bilateral_filter', 'd': 9, 'sigmaColor': 75, 'sigmaSpace': 75}
    args_nlm = {'type': 'nl_means', 'h': 10, 'templateWindowSize': 7, 'searchWindowSize': 21}
    smoothed = smooth(img_data, args_bf)

    threshold = 170
    thresholded = np.where(smoothed > threshold, smoothed, 0)
    # Display the image
    newC=[206, 206, 206]
    newscale=[0.0306, 0.0306, 0.0306]
    # plt.imshow(thresholded, extent=(1, newC[0]*newscale[0], 1, newC[1]*newscale[1]), aspect='auto')
    # plt.colorbar()
    # plt.show()

    # Load and display the MRI image
    mri_mat_contents = scipy.io.loadmat('./matlab_files/mri_mid_slice_image_data.mat')
    mid_slice_img_data = mri_mat_contents['mid_slice_img_data']
    newMRIC = mri_mat_contents['newMRIC'].flatten()  # Make sure you're using the right variable
    newscale_mri = mri_mat_contents['newscale'].flatten()  # Use a different variable to avoid confusion
    mid_slice_img_data = np.flip(mid_slice_img_data, axis=1)
    mid_slice_img_data *= 255/np.max(mid_slice_img_data)
    alpha = 2.0  # Contrast control (1.0 = original, >1.0 = higher contrast)
    beta = 0  # Brightness control (optional)
    mid_slice_img_data = sharpen(mid_slice_img_data)
    mid_slice_img_data = np.clip(alpha * mid_slice_img_data + beta, 0, 255)
    plt.imsave(r"C:\Users\awong\Downloads\mri_scan_head.png", np.flip(mid_slice_img_data, axis=0), cmap='gray')
    # Display the MRI image
    # plt.imshow(mid_slice_img_data, extent=(1, newMRIC[0]*newscale_mri[0], 1, newMRIC[1]*newscale_mri[1]), aspect='auto')
    # plt.colorbar()
    # plt.title('MRI Mid-Slice Image')
    # plt.gca().invert_yaxis()  # Match MATLAB's Y-axis direction
    # plt.show()

    row = 150
    col = 450

    # Rotate ultrasound scan
    rotated = rotate_image(thresholded, angle=-20)
    H, W = rotated.shape

    mid_slice_img_data[row:row+H, col:col+W] += rotated
    mid_slice_img_data = np.clip(mid_slice_img_data, 0, 255)
    # Display the superimposed image
    plt.imshow(mid_slice_img_data, extent=(1, newMRIC[0]*newscale_mri[0], 1, newMRIC[1]*newscale_mri[1]), aspect='auto')
    plt.colorbar()
    plt.title('MRI Mid-Slice Image')
    plt.gca().invert_yaxis()  # Match MATLAB's Y-axis direction
    plt.show()