import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import signal

# TODO:
#    
#   
#

def butterworthNotchFilter(d0, points, n1, n2, n, visualize=False):
    # Create a grid of frequency coordinates
    k1, k2 = np.meshgrid(np.arange(-round(n2/2)+1, math.floor(n2/2)+1), np.arange(-round(n1/2)+1, math.floor(n1/2)+1))
    
    # Compute the distance from the origin for each frequency coordinate
    d = np.sqrt(k1**2 + k2**2)
    
    # Initialize the filter with all ones
    h = np.ones_like(d)
    
    for point in points:
        # Extract the coordinates of the notch frequency
        u_k, v_k = point[0], point[1]
        
        # Swap u_k and v_k for frequency domain indexing
        u_k, v_k = n2 - u_k, n1 - v_k
        
        # Compute the distances from the notch frequency and its symmetric counterpart
        d1 = np.sqrt((k1 - u_k)**2 + (k2 - v_k)**2)
        d2 = np.sqrt((k1 + u_k)**2 + (k2 + v_k)**2)
        
        # Apply the notch filter response using the Butterworth formula
        h *= 1 / (1 + (d0**2) / (d1 * d2)**(2 * n))
    
    if visualize:
        visualize_filter(h)
    
    return h

# def notchfilter(img, center_freq, radius):
    
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


def visualize_filter(filter_mask):
    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(filter_mask, cmap="gray")

    a2 = fig.add_subplot(1,2,2, projection="3d")
    x,y = np.meshgrid(np.arange(filter_mask.shape[0]),np.arange(filter_mask.shape[1]))
    a2.plot_surface(x,y,filter_mask.T,cmap=plt.cm.coolwarm,linewidth=0, antialiased=False)
    plt.show()

def apply_filter(img,filter_mask):
    f = np.fft.fftshift(np.fft.fft2(img))
    f1 = f * filter_mask
    x1 = np.fft.ifft2(np.fft.ifftshift(f1))

    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(img,cmap="gray")
    ax1.set_title("Original")
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(abs(x1)/255, cmap="gray")
    ax2.set_title("Transformed")
    plt.show()


def notchFilter(image, center_frequency, bandwidth, phi):
    # convert center frequency and bandwidth to radians
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

    # Apply the notch filter using scipy.signal.lfilter2d
    filtered_image = signal.lfilter2d(numerator, denominator, image)

    return filtered_image



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
img_filtered =  notchFilter(img, freq ,10)

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