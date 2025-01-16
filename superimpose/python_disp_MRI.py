import scipy.io
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Load the .mat file
    mat_contents = scipy.io.loadmat('./matlab_files/mri_mid_slice_image_data.mat')
    mid_slice_img_data = mat_contents['mid_slice_img_data']
    newMRIC = mat_contents['newMRIC'].flatten()
    newscale = mat_contents['newscale'].flatten()

    # Display the image
    plt.imshow(mid_slice_img_data, extent=(1, newMRIC[0]*newscale[0], 1, newMRIC[1]*newscale[1]), aspect='auto')
    plt.colorbar()
    plt.title('MRI Mid-Slice Image')
    plt.gca().invert_yaxis()  # Match MATLAB's Y-axis direction
    plt.show()
