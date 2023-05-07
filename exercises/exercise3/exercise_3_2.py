import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# load image
img = cv.imread('images/birdie.jpg')

# get dimensions of image
dimensions = img.shape

# build meshgrid based on image dimensions
x = np.linspace(-1,1,dimensions[0])
y = np.linspace(-1,1,dimensions[1])

X, Y = np.meshgrid(x,y)

# build the 2D sine, adjusting X and Y in output, controls orientation
#   f(x,y) = sin(2*pi*x/lambda)
wavelength = 0.5  
amplitude = 1.0
sine2D = amplitude * np.sin(2*np.pi*Y/wavelength)

# compute 2d fft, centre it and compute magnitude spectrum
f = np.fft.fftshift(np.fft.fft2(sine2D))
magnitude_spectrum = 20*np.log(np.abs(f))

# get 1D slices

# is this code the same...
# middle_row = magnitude_spectrum[int(img.shape[0]/2),:]
# middle_column = magnitude_spectrum[int(img.shape[1]/2),:]

# as this code..
middle_row = magnitude_spectrum[int(img.shape[0]/2),:]
middle_column = magnitude_spectrum[:,int(img.shape[1]/2)]


# FFT 
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')
plt.axis('off')
plt.show()

# 1D slices
plt.subplot(1, 2, 1)
plt.plot(middle_row)
plt.title('Middle Row 1D Slice')

plt.subplot(1, 2,2)
plt.plot(middle_column)
plt.title('Middle Column 1D Slice')

plt.tight_layout()
plt.show()
