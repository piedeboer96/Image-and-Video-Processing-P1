import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import signal

"""
    Methods
"""

# helper method to visualize filter
def visualize_filter(filter_mask):
    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(filter_mask, cmap="gray")

    a2 = fig.add_subplot(1,2,2, projection="3d")
    x,y = np.meshgrid(np.arange(filter_mask.shape[0]),np.arange(filter_mask.shape[1]))
    a2.plot_surface(x,y,filter_mask.T,cmap=plt.cm.coolwarm,linewidth=0, antialiased=False)
    plt.show()

# based on the approach from the book
def butterworthNotchFilter(d0, center_frequency, bandwidth, n1, n2, n, visualize=False):
    
    # code adapted from lab 4 HPF
    k1, k2 = np.meshgrid(np.arange(-round(n2/2)+1, math.floor(n2/2)+1), np.arange(-round(n1/2)+1, math.floor(n1/2)+1))
    d = np.sqrt(k1**2 + k2**2)

    # the 'negative' frequency part
    d0_neg = -d0

    # the two high pass filters
    h1 = 1 / (1 + (d0 / d)**(2*n))
    h2 = 1 / (1 + (d0_neg / d)**(2*n))

    # take the order into account
    h = h1 * h2

    if visualize:
        visualize_filter(h)

    return h

# add periodic noise to an image
def add_periodic_noise(img, amplitude=1.5, frequency=33):

    # grayscale 
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    dimensions = img.shape

    print('dimensions', dimensions)

    # build meshgrid based on image dimensions
    x = np.linspace(-1,1,dimensions[1])
    y = np.linspace(-1,1,dimensions[0])

    X, Y = np.meshgrid(x,y)

    # build the 2D sine, adjusting X and Y in output controls orientation  
    wavelength = 1/frequency
    sine2D = amplitude * np.sin(2*np.pi*Y/wavelength)
    img_noisy = img + sine2D

    return img_noisy

    # now add the noise


# load image
img = cv.imread('images/birdie.jpg')

# frequency for noise and filter
freq = 30
amplitude = 10
radius = 10

# add noise to the image
img_noisy = add_periodic_noise(img, amplitude, freq)

# get magnitude spectrum of noisy image
f = np.fft.fft2(img_noisy)
fshift = np.fft.fftshift(f)
magnitude_spectrum_img_noisy = 20*np.log(np.abs(fshift))

# # apply our filter notch filter
img_filtered =  butterworthNotchFilter(img, freq ,10)

# get magnitude spectrum of fitlered image
f = np.fft.fft2(img_filtered)
fshift = np.fft.fftshift(f)
magnitude_spectrum_filtered_img = 20*np.log(np.abs(fshift))


# TODO;
#
#   Create a frequency domain filter to remove the periodic noise from the previous question’s noisy image. Display and discuss: 
#   (1) the de-noising filter’s FT magnitude in 1D and 2D, 
#   (2) the de-noised image’s FT magnitude in 1D and 2D, 
#   (3) the resulting de-noised image.
#
#


# plots
plt.subplot(2,2,1)
plt.imshow(img_noisy,  cmap='Greys')
plt.title('Noisy Image')
plt.axis('off')

# plt.subplot(2,2,2)
# plt.imshow(magnitude_spectrum_img_noisy,cmap='Greys')
# plt.title('Noisy Image Magnitude Spectrum')
# plt.axis('off')

# plt.subplot(2,2,3)
# plt.imshow(img_filtered,cmap='Greys')
# plt.title('Filtered Image')
# plt.axis('off')

# plt.subplot(2,2,4)
# plt.imshow(magnitude_spectrum_filtered_img,cmap='Greys')
# plt.title('Magnitude Spectrum Filtered Image')
# plt.axis('off')

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




### apply butterworth LPF
def butterworthLPF(d0,n1,n2,n, visualize = False):
    k1,k2 = np.meshgrid(np.arange(-round(n2/2)+1, math.floor(n2/2)+1), np.arange(-round(n1/2)+1, math.floor(n1/2)+1))
    d = np.sqrt(k1**2 + k2**2)
    h = 1 / (1 + (d / d0)**(2*n))
    if visualize:
        visualize_filter(h)
    return h 
    
print('img noisy...', img_noisy.shape)
f = np.fft.fftshift(np.fft.fft2(img_noisy))

img = img_noisy
d0 = round(img.shape[0] /20)
h = butterworthLPF(d0,img.shape[0],img.shape[1],2);
f1 = f*h

x1 = abs(np.fft.ifft2(np.fft.ifftshift(f1)))
x1[x1==0] = 1e-2

x2 = img / x1

fig = plt.figure(figsize=(20,8))
ax1 = fig.add_subplot(1,3,1)
ax1.imshow(img,cmap="gray")
ax1.set_title("Original")
ax2 = fig.add_subplot(1,3,2)
ax2.imshow(abs(x1)/255, cmap="gray")
ax2.set_title("Blurred")
ax2 = fig.add_subplot(1,3,3)
ax2.imshow(abs(x2), cmap="gray")
ax2.set_title("Pencil Sketch")
plt.show()



 


print('img noisy...', img_noisy.shape)
f = np.fft.fftshift(np.fft.fft2(img_noisy))

# img = img_noisy
# d0 = round(img.shape[0] /20)
# h = nott(d0,img.shape[0],img.shape[1],1,visualize=True);
# f1 = f*h


# TODO: apply inverse fft to f1 , 
#       and display both the noisy image and notch-filtered image

filtered_img = np.fft.ifft2(np.fft.ifftshift(f1)).real

# Display the noisy image and notch-filtered image
plt.subplot(1, 2, 1)
plt.imshow(img_noisy, cmap='gray')
plt.title('Noisy Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filtered_img, cmap='gray')
plt.title('Notch-Filtered Image')
plt.axis('off')

plt.show()