import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from utils import *
import cv2

if __name__ == '__main__':
    # Load the .mat file
    mat_contents = scipy.io.loadmat('./matlab_files/mid_saggital_image_data.mat')
    #print ("Matrix Contents: ", mat_contents)
    img_data = mat_contents['mid_sag_img_data']
    print ("Image Data: ", img_data.shape)

    newC=[206, 206, 206]
    newscale=[0.0306, 0.0306, 0.0306]

    # Apply bilateral filter
    d = 9              # Diameter of the pixel neighborhood
    sigmaColor = 75    # Filter sigma in color space
    sigmaSpace = 75    # Filter sigma in coordinate space
    img_data = cv2.bilateralFilter(img_data, d, sigmaColor, sigmaSpace)
    # Apply a threshold to mask the blue regions
    threshold = np.percentile(img_data, 95)  # Adjust this percentile to isolate yellow regions
    masked_img_data = np.where(img_data > threshold, img_data, np.nan)

    print(masked_img_data.shape)

    # Display the image
    #plt.imshow(img_data, extent=(1, newC[0]*newscale[0], 1, newC[1]*newscale[1]), aspect='auto')
    plt.imshow(masked_img_data, extent=(1, newC[0]*newscale[0], 1, newC[1]*newscale[1]), aspect='auto')
    plt.colorbar()
    plt.show()
