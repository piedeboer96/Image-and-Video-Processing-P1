import cv2
import numpy as np
import matplotlib.pyplot as plt

# method to build 2D adapted from:
#   https://thepythoncodingbook.com/2022/05/28/numpy-meshgrid/

def build_2D_sine(img, amplitude, frequency):

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

    # now add the noise

# Load the image
img = cv2.imread('images/birdie.jpg', cv2.IMREAD_GRAYSCALE)
X,Y, sine = build_2D_sine(img,amplitude=10,frequency=100)

# Display the 2D cosine or sine
plt.imshow(sine, cmap='gray')
plt.title('2D sine')
plt.show()

f = np.fft.fft2(sine)
fshift = np.fft.fftshift(f)

# Compute the magnitude spectrum
magnitude_spectrum = np.abs(fshift)
magnitude_spectrum = np.log(magnitude_spectrum + 1)

# Display the magnitude spectrum
plt.imshow(magnitude_spectrum)
plt.title('Magnitude Spectrum')
plt.show()

# Compute the middle row and column of the magnitude spectrum
middle_row = magnitude_spectrum[magnitude_spectrum.shape[0]//2,:]
middle_col = magnitude_spectrum[:,magnitude_spectrum.shape[1]//2]

# Display the middle row and column
plt.plot(middle_row)
plt.title('Middle Row of FFT Magnitude')
plt.show()

plt.plot(middle_col)
plt.title('Middle Column of FFT Magnitude')
plt.show()







