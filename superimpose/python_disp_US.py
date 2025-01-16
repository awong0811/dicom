import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from utils import *
import cv2

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
    plt.imshow(thresholded, extent=(1, newC[0]*newscale[0], 1, newC[1]*newscale[1]), aspect='auto')
    plt.colorbar()
    plt.show()
