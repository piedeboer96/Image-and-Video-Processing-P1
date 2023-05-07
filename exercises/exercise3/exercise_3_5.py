import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# butterworthnotch reject transfer function (Gonzalez, page 300)
def H_NR(u, v, d0, n):
    M, N = u.shape[0], u.shape[1]
    D0 = np.sqrt((u-M/2)**2 + (v-N/2)**2)
    D = np.zeros((M, N, 3))
    D[:, :, 0] = np.sqrt((u-M/2)**2 + (v-N/2-2)**2)
    D[:, :, 1] = np.sqrt((u-M/2+2)**2 + (v-N/2)**2)
    D[:, :, 2] = np.sqrt((u-M/2-2)**2 + (v-N/2)**2)
    H_NR = 1
    for k in range(3):
        numerator = (d0**2)/(D0*D[:, :, k])
        H_NR *= 1/(1 + numerator**n)*1/(1 + (d0/D[:, :, k])**n)
    return H_NR

# method to add periodic noise to image
def add_periodic_noise(img, amplitude, frequency):

    # grayscale
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

    # now add the noise
    return np.add(img, sine2D)

# load image
img = cv.imread('images/birdie.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# image with added periodic noise
noisy_img = add_periodic_noise(img, amplitude=10, frequency=2)

# compute the FFT of the noisy image
F = np.fft.fftshift(np.fft.fft2(noisy_img))

# apply the notch-reject filter
# 

d0 = 3     # control widht
n = 2
H = H_NR(*np.meshgrid(np.arange(F.shape[0]), np.arange(F.shape[1]), indexing='ij'), d0=d0, n=n)
G = F * H

# compute the inverse FFT to obtain the filtered image
g = np.fft.ifft2(np.fft.ifftshift(G)).real



# loop through different orders of the filter and plot the filtered images
# orders = [8]
# fig, axs = plt.subplots(1, len(orders)+1, figsize=(20, 5))
# axs[0].imshow(noisy_img, cmap='gray')
# axs[0].set_title('Noisy image')
# axs[0].axis('off')
# for i, n in enumerate(orders):
#     H = H_NR(*np.meshgrid(np.arange(F.shape[0]), np.arange(F.shape[1]), indexing='ij'), d0=d0, n=n)
#     G = F * H
#     g = np.fft.ifft2(np.fft.ifftshift(G)).real
#     axs[i+1].imshow(g, cmap='gray')
#     axs[i+1].set_title('Filtered image, order={}'.format(n))
#     axs[i+1].axis('off')
# plt.show()

# # compute the notch-reject filter with different radius values
# radius_list = [3]
# n = 16

# fig, axs = plt.subplots(1, len(radius_list), figsize=(15, 5))

# for i, radius in enumerate(radius_list):
#     H = H_NR(*np.meshgrid(np.arange(F.shape[0]), np.arange(F.shape[1]), indexing='ij'), d0=radius, n=n)
#     G = F * H
#     g = np.fft.ifft2(np.fft.ifftshift(G)).real
#     axs[i].imshow(g, cmap='gray')
#     axs[i].set_title('Filtered image\nRadius: {}'.format(radius))
#     axs[i].axis('off')

# plt.show()


# TODO: test image with amplitude added 25 and amplitude 250 , both for frequency 2 hz
#    use the notch reject filter with freq=2

# this is the code for 3.5



# test image with added periodic noise
noisy_img_25 = add_periodic_noise(img, amplitude=25, frequency=2)
noisy_img_250 = add_periodic_noise(img, amplitude=750, frequency=2)

# compute the FFT of the noisy images
F_25 = np.fft.fftshift(np.fft.fft2(noisy_img_25))
F_250 = np.fft.fftshift(np.fft.fft2(noisy_img_250))

# apply the notch-reject filter with frequency 2 and different amplitudes
radius = 3
n = 16
H = H_NR(*np.meshgrid(np.arange(F_25.shape[0]), np.arange(F_25.shape[1]), indexing='ij'), d0=radius, n=n)

G_25 = F_25 * H
G_250 = F_250 * H

g_25 = np.fft.ifft2(np.fft.ifftshift(G_25)).real
g_250 = np.fft.ifft2(np.fft.ifftshift(G_250)).real

# plot the results

plt.subplot(2, 2, 1); plt.imshow(noisy_img_25,cmap='gray'); plt.title('Noisy Image amplitude 25'); plt.axis('off')
plt.subplot(2, 2, 2); plt.imshow(noisy_img_250,cmap='gray'); plt.title('Noisy Image amplitude 250'); plt.axis('off')
plt.subplot(2, 2, 3); plt.imshow(g_25,cmap='gray'); plt.title('Filtered Image amplitude 25'); plt.axis('off')
plt.subplot(2, 2, 4); plt.imshow(g_250,cmap='gray'); plt.title('Filtered Image amplitude 250'); plt.axis('off')

plt.show()