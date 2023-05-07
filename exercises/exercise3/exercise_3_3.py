import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# method to add periodic noise to image
def add_periodic_noise_spatial(img, amplitude, frequency):

    # grayscale
    dimensions = img.shape

    print('dimensions', dimensions)

    # build meshgrid based on image dimensions
    x = np.linspace(-1,1,dimensions[1])
    y = np.linspace(-1,1,dimensions[0])

    X, Y = np.meshgrid(x,y)

    # build the 2D sine, adjusting X and Y in output, controls orientation
    #   f(x,y) = sin(2*pi*x/lambda)
    
    wavelength = 1/frequency

    sine2D = amplitude * np.sin(2*np.pi*Y/wavelength)

    # now add the noise
    return np.add(img, sine2D)

# load image
img = cv.imread('images/birdie.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# image with added periodic noise
img_corrupt = add_periodic_noise_spatial(img, amplitude=100, frequency=2)

""" Magnitude spectra """

# original
f1 = np.fft.fft2(img)
fshift1 = np.fft.fftshift(f1)
magnitude_spectrum_img = np.abs(fshift1)
magnitude_spectrum_img = np.log(magnitude_spectrum_img + 1)

# noisy image
f2 = np.fft.fft2(img_corrupt)
fshift2 = np.fft.fftshift(f2)
magnitude_spectrum_corrupt = np.abs(fshift2)
magnitude_spectrum_corrupt = np.log(magnitude_spectrum_corrupt + 1)

""" 1D Slices """
# middle row and middle colum of original image
middle_row_img = magnitude_spectrum_img[magnitude_spectrum_img.shape[0]//2,:]
middle_col_img = magnitude_spectrum_img[:,magnitude_spectrum_img.shape[1]//2]

# get 1D slice middle row and column 'corrupt'
middle_row_corrupt = magnitude_spectrum_corrupt[magnitude_spectrum_corrupt.shape[0]//2,:]
middle_column_corrupt = magnitude_spectrum_corrupt[:,magnitude_spectrum_corrupt.shape[1]//2]

""" Plots """
# plot original and corrupt image
plt.subplot(1, 2, 1)
plt.imshow(img,cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_corrupt, cmap='gray')
plt.title('Noisy Image')
plt.axis('off')

plt.tight_layout()
plt.show()

# plot magnitude spectrum corrupt image
plt.subplot(1,2,1)
plt.imshow(magnitude_spectrum_corrupt)
plt.title('Magnitude Spectrum Corrupt Image')
plt.axis('off')

plt.imshow(magnitude_spectrum_corrupt)
plt.title('Magnitude Spectrum Corrupt Image')
plt.axis('off')

plt.tight_layout()
plt.show()

plt.plot(middle_row_corrupt)
plt.title('Middle Row 1D Slice Corrupt')
plt.ylabel('Magnitude')
plt.show()

plt.plot(middle_column_corrupt)
plt.title('Middle Column 1D Slice Corrupt')
plt.ylabel('Magnitude')
plt.show()


