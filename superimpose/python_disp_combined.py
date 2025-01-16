import scipy.io
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    # Load and display the ultrasound (US) image
    us_mat_contents = scipy.io.loadmat('./matlab_files/mid_saggital_image_data.mat')
    img_data = us_mat_contents['mid_sag_img_data']

    newC = [206, 206, 206]  # You might want to ensure these are correct for your dataset
    newscale = [0.0306, 0.0306, 0.0306]

    # Apply a threshold to mask the blue regions (low values)
    threshold = np.percentile(img_data, 95)  # Adjust this percentile to isolate yellow regions
    masked_img_data = np.where(img_data > threshold, img_data, np.nan)

    # Display the US image
    plt.imshow(masked_img_data, extent=(1, newC[0]*newscale[0], 1, newC[1]*newscale[1]), aspect='auto')
    plt.colorbar()
    plt.title('Ultrasound Mid-Sagittal Image')
    plt.show()

    # Load and display the MRI image
    mri_mat_contents = scipy.io.loadmat('./matlab_files/mri_mid_slice_image_data.mat')
    mid_slice_img_data = mri_mat_contents['mid_slice_img_data']
    newMRIC = mri_mat_contents['newMRIC'].flatten()  # Make sure you're using the right variable
    newscale_mri = mri_mat_contents['newscale'].flatten()  # Use a different variable to avoid confusion

    # Display the MRI image
    plt.imshow(mid_slice_img_data, extent=(1, newMRIC[0]*newscale_mri[0], 1, newMRIC[1]*newscale_mri[1]), aspect='auto')
    plt.colorbar()
    plt.title('MRI Mid-Slice Image')
    plt.gca().invert_yaxis()  # Match MATLAB's Y-axis direction
    plt.show()
