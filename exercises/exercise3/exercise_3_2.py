import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# load image
img = cv.imread('images/birdie.jpg', cv.IMREAD_GRAYSCALE)

""" code for 2d sine meshgrid """

# get dimensions of image
dimensions = img.shape

print('dimensions', dimensions)

# build meshgrid based on image dimensions
x = np.linspace(-1,1,dimensions[0])
y = np.linspace(-1,1,dimensions[1])

X, Y = np.meshgrid(x,y)

# build the 2D sine, adjusting X and Y in output, controls orientation
#   f(x,y) = sin(2*pi*x/lambda)
wavelength = 0.3  
amplitude = 1.5
sine2D = amplitude * np.sin(2*np.pi*Y/wavelength)

# compute 2d fft, centre it and compute magnitude spectrum
f = np.fft.fft2(sine2D)
fshift = np.fft.fftshift(sine2D)
magnitude_spectrum = 20*np.log(np.abs(fshift))

print(magnitude_spectrum.shape)


# get 1D slices
middle_row = magnitude_spectrum[magnitude_spectrum.shape[0] // 2, :]
middle_column = magnitude_spectrum[:, magnitude_spectrum.shape[1] // 2]

# plots
plt.subplot(1, 3, 1)
plt.imshow(magnitude_spectrum)
plt.title('Magnitude Spectrum')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.plot(middle_row)
plt.title('Middle Row 1D Slice')
# plt.xlabel('Index')
plt.ylabel('Magnitude')

plt.subplot(1, 3, 3)
plt.plot(middle_column)
plt.title('Middle Column 1D Slice')
# plt.xlabel('Index')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()



# TODO:
#   change x-axis to frequency 


# DESCRIPTION:
#   make sense... cause one is the actual 'amplitude' 
#   other isn't depending on which slice