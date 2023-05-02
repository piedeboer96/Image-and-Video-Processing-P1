import numpy as np
import scipy.signal as signal
import cv2 as cv
import matplotlib.pyplot as plt


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

    return img + sine2D

    # now add the noise

def notchFilter(image, center_frequency, bandwidth, phi):
    
    # bring image into frequency domain
    f = np.fft.fftshift(np.fft.fft2(img))
    
    # Convert center frequency and bandwidth to radians
    omega0 = 2 * np.pi * center_frequency
    bandwidth_rad = 2 * np.pi * bandwidth

    # Compute the notch frequencies
    f1 = omega0 - bandwidth_rad / 2
    f2 = omega0 + bandwidth_rad / 2

    # Compute the complex conjugate zeros
    zero1 = np.exp(1j * f1)
    zero2 = np.exp(1j * f2)

    # Compute the numerator and denominator coefficients of the transfer function
    numerator = [1, -2 * np.cos(omega0), 1]
    denominator = [1, -2 * np.real(zero1 * np.exp(1j * phi)), np.abs(zero1)**2]

    # Apply the filter to the image
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    filtered_image = np.multiply(fshift, np.polyval(numerator, zero1 * np.exp(1j * phi)) / np.polyval(denominator, zero1 * np.exp(1j * phi)))

    return filtered_image



# super notch filter



# load image
img = cv.imread('images/beach.jpg')

# add noise
freq = 30
amplitude = 10
img_noisy = add_periodic_noise(img,freq,amplitude)

# apply notch filter
bandwith = 10
phase = 0
# img_cleaned = notchFilter(img_noisy, freq, bandwith,phase)


# show everything
plt.subplot(2,2,1)
plt.imshow(img_noisy,  cmap='Greys')
plt.title('Noisy Image')
plt.axis('off')

# plt.subplot(2,2,2)
# plt.imshow(img_cleaned,cmap='Greys')
# plt.title('Noisy Image Magnitude Spectrum')
# plt.axis('off')

plt.tight_layout()
plt.show()