import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc


# method that returns image with periodic noise added 
def add_periodic_noise(img, amplitude=1.5, frequency=33):

    # grayscale
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

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


img = cv.imread('images/birdie.jpg')

img_original = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# image with added periodic noise
img_corrupt = add_periodic_noise(img, 50, 60)

# original magnitude spectrum
f = np.fft.fft2(img_original)
fshift = np.fft.fftshift(f)
magnitude_spectrum_original = 20*np.log(np.abs(fshift))

# corrupt image magnitude spectrum
f = np.fft.fft2(img_corrupt)
fshift = np.fft.fftshift(f)
magnitude_spectrum_corrupt = 20*np.log(np.abs(fshift))

# get 1D slice middle row and column 'original'
middle_row_original = magnitude_spectrum_original[int(img.shape[0]/2),:]
middle_column_original = magnitude_spectrum_original[:,int(img.shape[1]/2)]

# get 1D slice middle row and column 'corrupt'
middle_row_corrupt = magnitude_spectrum_corrupt[int(img.shape[0]/2),:]
middle_column_corrupt = magnitude_spectrum_corrupt[:,int(img.shape[1]/2)]

# plot original and corrupt image
plt.subplot(1, 2, 1)
plt.imshow(img_original, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_corrupt, cmap='gray')
plt.title('Noisy Image')
plt.axis('off')

plt.tight_layout()
plt.show()

# plot magnitude spectrum original and corrupt image
plt.subplot(1, 2, 1)
plt.imshow(magnitude_spectrum_original, cmap='gray')
plt.title('Magnitude Spectrum Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(magnitude_spectrum_corrupt, cmap='gray')
plt.title('Magnitude Spectrum Corrupt Image')
plt.axis('off')

plt.tight_layout()
plt.show()

plt.subplot(1, 2, 1)
plt.plot(middle_row_original)
plt.title('Middle Row 1D Slice Original')
plt.ylabel('Magnitude')

plt.subplot(1, 2, 2)
plt.plot(middle_column_original)
plt.title('Middle Column 1D Slice Original')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()

plt.subplot(1, 2, 1)
plt.plot(middle_row_corrupt)
plt.title('Middle Row 1D Slice Corrupt')
plt.ylabel('Magnitude')

plt.subplot(1, 2, 2)
plt.plot(middle_column_corrupt)
plt.title('Middle Column 1D Slice Corrupt')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show
