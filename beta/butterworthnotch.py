import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

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
    h = 1
    i = 1
    while i < n:
        h *= h1 * h2
        i += 1

    if visualize:
        visualize_filter(h)

    return h

# add periodic noise to an image
def add_periodic_noise(img, amplitude=1.5, frequency=33):

    # grayscale 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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


"""
    Main Code
"""

# load image and add noise
image = cv2.imread('images/birdie.jpg')  
image_noisy = add_periodic_noise(image,amplitude=10,frequency=33)

# define the notch filter parameters
center_frequency = 30; bandwidth = 10; order = 5

# use framework from lab 4
img = image_noisy
f = np.fft.fftshift(np.fft.fft2(image_noisy))

d0 = round(img.shape[0] /20)
h = butterworthNotchFilter(d0,center_frequency=33,bandwidth=bandwidth,n1=img.shape[0],n2=img.shape[1],n=order, visualize=True)

butter_mask = h

apply_filter(image_noisy, butter_mask)


f1 = f*h

x1 = abs(np.fft.ifft2(np.fft.ifftshift(f1)))
x1[x1==0] = 1e-2

# TODO: bring back into normal grayscale the image after applying notch filter and plot it !


# fig = plt.figure(figsize=(20,8))
# ax1 = fig.add_subplot(1,3,1)
# ax1.imshow(image_noisy,cmap="gray")
# ax1.set_title("Original")
# ax2 = fig.add_subplot(1,3,2)
# ax2.imshow(abs(x1)/255, cmap="gray")
# ax2.set_title("Notch")
# plt.show()

# bring back into normal grayscale the image after applying notch filter
img_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(f1)))

# plot the filtered image
fig = plt.figure(figsize=(20,8))
ax1 = fig.add_subplot(1,3,1)
ax1.imshow(image_noisy,cmap="gray")
ax1.set_title("Original")
ax2 = fig.add_subplot(1,3,2)
ax2.imshow(abs(x1)/255, cmap="gray")
ax2.set_title("Notch")
ax3 = fig.add_subplot(1,3,3)
ax3.imshow(img_filtered,cmap="gray")
ax3.set_title("Filtered")
plt.show()