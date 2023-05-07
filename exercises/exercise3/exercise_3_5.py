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
noisy_img = add_periodic_noise(img, amplitude=100, frequency=2)

# compute the FFT of the noisy image
F = np.fft.fftshift(np.fft.fft2(noisy_img))


d0 = 3     # control widht
n = 8
H = H_NR(*np.meshgrid(np.arange(F.shape[0]), np.arange(F.shape[1]), indexing='ij'), d0=d0, n=n)
G = F * H

# compute the inverse FFT to obtain the filtered image
g = np.fft.ifft2(np.fft.ifftshift(G)).real

""" ORDER AND RADIUS """

# loop through different orders of the filter and plot the filtered images
# d0=3
# orders = [2,4,6,8]
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

# # # compute the notch-reject filter with different radius values
# radius_list = [1,3,5,10]
# n = 8


# fig, axs = plt.subplots(1, len(radius_list), figsize=(15, 5))

# for i, radius in enumerate(radius_list):
#     H = H_NR(*np.meshgrid(np.arange(F.shape[0]), np.arange(F.shape[1]), indexing='ij'), d0=radius, n=n)
#     G = F * H
#     g = np.fft.ifft2(np.fft.ifftshift(G)).real
#     axs[i].imshow(g, cmap='gray')
#     axs[i].set_title('Filtered image\nRadius: {}'.format(radius))
#     axs[i].axis('off')

# plt.show()


""" INCREASED AMPLITUDE """
# this is the code for 3.5

# test image with added periodic noise
noisy_img_25 = add_periodic_noise(img, amplitude=25, frequency=2)
noisy_img_250 = add_periodic_noise(img, amplitude=1500, frequency=2)

# compute the FFT of the noisy images
F_25 = np.fft.fftshift(np.fft.fft2(noisy_img_25))
F_250 = np.fft.fftshift(np.fft.fft2(noisy_img_250))

# apply the notch-reject filter with frequency 2 and different amplitudes
radius = 3
n = 8
H = H_NR(*np.meshgrid(np.arange(F_25.shape[0]), np.arange(F_25.shape[1]), indexing='ij'), d0=radius, n=n)

G_25 = F_25 * H
G_250 = F_250 * H

g_25 = np.fft.ifft2(np.fft.ifftshift(G_25)).real
g_250 = np.fft.ifft2(np.fft.ifftshift(G_250)).real

# plot the results

# plt.subplot(2, 2, 1); plt.imshow(noisy_img_25,cmap='gray'); plt.title('Noisy Image amplitude 25'); plt.axis('off')
# plt.subplot(2, 2, 2); plt.imshow(noisy_img_250,cmap='gray'); plt.title('Noisy Image amplitude 250'); plt.axis('off')
# plt.subplot(2, 2, 3); plt.imshow(g_25,cmap='gray'); plt.title('Filtered Image amplitude 25'); plt.axis('off')
# plt.subplot(2, 2, 4); plt.imshow(g_250,cmap='gray'); plt.title('Filtered Image amplitude 500'); plt.axis('off')

# plt.show()

""" 1D slices of noisy and filtered for different amplitudes """

# magnitude spectra of noisy
F_25 = np.fft.fftshift(np.fft.fft2(noisy_img_25))
fshift2 = np.fft.fftshift(F_25)
ms_noisy_25 = np.abs(fshift2)
ms_noisy_25 = np.log(ms_noisy_25 + 1)

F_250 = np.fft.fftshift(np.fft.fft2(noisy_img_250))
fshift2 = np.fft.fftshift(F_250)
ms_noisy_250 = np.abs(fshift2)
ms_noisy_250 = np.log(ms_noisy_25 + 1)

# magnitude spectra of filtered
fshift25 = np.fft.fftshift(G_25)
ms_filtered25 = np.abs(fshift25)  
ms_filtered25 = np.log(ms_filtered25 + 1) 

fshift500 = np.fft.fftshift(G_250)
ms_filtered500 = np.abs(fshift500)  
ms_filtered500 = np.log(ms_filtered500 + 1) 

""" 1D Slices """
# noisy 25
middle_row_noisy_25 = ms_noisy_25[ms_noisy_25.shape[0]//2,:]
middle_column_noisy_25 = ms_noisy_25[:,ms_noisy_25.shape[1]//2]

# noisy 250
middle_row_noisy_250 = ms_noisy_250[ms_noisy_250.shape[0]//2,:]
middle_column_noisy_250 = ms_noisy_250[:,ms_noisy_250.shape[1]//2]

# denoised 25
middle_row_denoise_25 = ms_filtered25[ms_filtered25.shape[0]//2,:]
middle_column_denoise_25 = ms_filtered25[:,ms_filtered25.shape[1]//2]

# denoised 250
middle_row_denoise_500 = ms_filtered500[ms_filtered500.shape[0]//2,:]
middle_column_denoise_500 = ms_filtered500[:,ms_filtered500.shape[1]//2]

# provide 4 plots containing the following subplots 
#   plot 1:  magnitude spectra 2D noisy amplitude=25 and amplitude=250
#   plot 2:  magnitude spectra 2D filterd/denoised amplitude=25 and amplitude=250
#   plot 3:  1d slices of noisy , amplitude=25 and amplitude=250
#   plot 4:  1d slices of filtered/denoised, amplitude=25 and amplitude=250


# plt.subplot(1,2,1)
# plt.title('2D magntiude spectrum low noise level')
# plt.imshow(ms_noisy_25)
# plt.subplot(1,2,2)
# plt.title('2D magntiude spectrum high noise level')
# plt.imshow(ms_noisy_250)
# plt.tight_layout()
# plt.colorbar()
# plt.show()

# plt.subplot(1,2,1)
# plt.title('2D magntiude spectrum denoised (low)')
# plt.imshow(ms_filtered25)
# plt.subplot(1,2,2)
# plt.title('2D magntiude spectrum denoised (high)')
# plt.imshow(ms_filtered500)
# plt.tight_layout()
# plt.colorbar()
# plt.show()

plt.subplot(1,2,1)
plt.title('1D middle col noisy (low)')
plt.plot(middle_row_noisy_25)
plt.subplot(1,2,2)
plt.title('1D middle col noisy (high)')
plt.plot(middle_row_noisy_250)
plt.tight_layout()
plt.show()

plt.subplot(1,2,1)
plt.title('1D middle row denoise (low)')
plt.plot(middle_row_denoise_25)
plt.subplot(1,2,2)
plt.title('1D middle row denoise (high)')
plt.plot(middle_row_denoise_500)
plt.tight_layout()
plt.show()