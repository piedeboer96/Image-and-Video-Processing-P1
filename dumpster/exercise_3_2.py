import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# method to build meshgrid 2D sine based on image dimensions
def build_2D_sine(img, amplitude=1.5, frequency=33):

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

    return X, Y, sine2D

# load image
img = cv.imread('images/birdie.jpg', cv.COLOR_BGR2GRAY)

# add periodic noise
X,Y, sine2D = build_2D_sine(img,10,33)

# compute 2D FFT of my sinusoid meshgrid
sine2D_fft = np.fft.fft2(sine2D)

# shift the DC component to the center of the spectrum
sine2D_fft_shifted = np.fft.fftshift(sine2D_fft)

# compute the magnitude spectrum
magnitude_spectrum = np.abs(sine2D_fft_shifted)
magnitude_spectrum = np.log(magnitude_spectrum + 1)

# display the magnitude spectrum
plt.imshow(magnitude_spectrum)
plt.show()

# get 1D slices
m_row = int(img.shape[0]/2)
m_col = int(img.shape[1]/2)

slice_m_row = magnitude_spectrum[m_row,:]
slice_m_col = magnitude_spectrum[:,m_col]

# FFT 
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')
plt.axis('off')
plt.show()

# 1D slices
plt.subplot(1, 2, 1)
plt.plot(slice_m_row)
plt.title('Middle Row 1D Slice')

plt.subplot(1, 2,2)
plt.plot(slice_m_col)
plt.title('Middle Column 1D Slice')

plt.tight_layout()
plt.show()
