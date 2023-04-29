import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def notchfilter(img, center_freq, radius):
    
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # bring img into frequency domain using fft
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # Compute dimensions of the image
    rows = img.shape[0]
    cols = img.shape[1]

    # Create a notch filter mask
    mask = np.ones((rows, cols), np.uint8)

    # Compute the center coordinates of the notch filter
    center_x = center_freq
    center_y = center_freq

    # Create a circular notch in the filter mask
    cv.circle(mask, (center_x, center_y), radius, 0, -1)

    # Apply filter in the frequency domain
    filtered_fshift = fshift * mask

    # Perform inverse Fourier transform
    filtered_f = np.fft.ifftshift(filtered_fshift)
    filtered_img = np.abs(np.fft.ifft2(filtered_f))

    return filtered_img

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



# load image
img = cv.imread('images/birdie.jpg')



# frequency for noise and filter
freq = 30

# add noise to the image
img_noisy = add_periodic_noise(img, 20, freq)

# get magnitude spectrum of noisy image
f = np.fft.fft2(img_noisy)
fshift = np.fft.fftshift(f)
magnitude_spectrum_img_noisy = 20*np.log(np.abs(fshift))

# # apply our filter notch filter
img_filtered = notchfilter(img, freq ,10)

# get magnitude spectrum of fitlered image
f = np.fft.fft2(img_filtered)
fshift = np.fft.fftshift(f)
magnitude_spectrum_filtered_img = 20*np.log(np.abs(fshift))


# TODO;
#
#   Create a frequency domain filter to remove the periodic noise from the previous question’s noisy image. Display and discuss: 
#   (1) the de-noising filter’s FT magnitude in 1D and 2D, 
#   (2) the de-noised image’s FT magnitude in 1D and 2D, (3) the resulting de-noised image.
#
#


# plots
plt.subplot(2,2,1)
plt.imshow(img_noisy,  cmap='Greys')
plt.title('Noisy Image')
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(magnitude_spectrum_img_noisy,cmap='Greys')
plt.title('Noisy Image Magnitude Spectrum')
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(img_filtered,cmap='Greys')
plt.title('Filtered Image')
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(magnitude_spectrum_filtered_img,cmap='Greys')
plt.title('Magnitude Spectrum Filtered Image')
plt.axis('off')

plt.tight_layout()
plt.show()


# Plot 1D magnitude spectrum along a row
plt.figure(figsize=(8, 4))
row_idx = magnitude_spectrum_filtered_img.shape[0] // 2  # Select middle row
plt.plot(magnitude_spectrum_filtered_img[row_idx, :], color='blue')
plt.title('1D Magnitude Spectrum along a Row')
plt.xlabel('Column')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()

# Plot 1D magnitude spectrum along a column
plt.figure(figsize=(8, 4))
col_idx = magnitude_spectrum_filtered_img.shape[1] // 2  # Select middle column
plt.plot(magnitude_spectrum_filtered_img[:, col_idx], color='red')
plt.title('1D Magnitude Spectrum along a Column')
plt.xlabel('Row')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()