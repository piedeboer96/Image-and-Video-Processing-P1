import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

# TODO:
#   improve the add periodic noise method to work with
#   frequencies as input argument... :)

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

    return np.add(img, sine2D)

    # now add the noise


img = cv.imread('images/birdie.jpg')

# image with added periodic noise
img_corrup = add_periodic_noise(img,15,0.3)

# calculate fft, centre and display magnitude
f = np.fft.fft2(img_corrup)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

# get 1D slide middle row and column
middle_row = magnitude_spectrum[magnitude_spectrum.shape[0] // 2, :]
middle_column = magnitude_spectrum[:, magnitude_spectrum.shape[1] // 2]

plt.subplot(2, 2, 1)
plt.imshow(img_corrup, cmap='Grays')
plt.title('Noisy Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(magnitude_spectrum)
plt.title('Magnitude Spectrum')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.plot(middle_row)
plt.title('Middle Row 1D Slice')
# plt.xlabel('Index')
plt.ylabel('Magnitude')

plt.subplot(2, 2, 4)
plt.plot(middle_column)
plt.title('Middle Column 1D Slice')
# plt.xlabel('Index')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()

