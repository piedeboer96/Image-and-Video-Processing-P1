import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

# visualize filter
def visualize_filter(filter_mask):
    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(filter_mask, cmap="gray")

    a2 = fig.add_subplot(1,2,2, projection="3d")
    x,y = np.meshgrid(np.arange(filter_mask.shape[0]),np.arange(filter_mask.shape[1]))
    a2.plot_surface(x,y,filter_mask.T,cmap=plt.cm.coolwarm,linewidth=0, antialiased=False)
    plt.show()

# periodic noise 
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

    sine2D = amplitude * np.sin(2*np.pi*X/wavelength)

    # now add the noise
    return np.add(img, sine2D)

# ideal bandreject 
def idealbandreject(n1, n2, freq1, freq2, visualize=False):
    
    k1, k2 = np.meshgrid(np.arange(-round(n2/2)+1, math.floor(n2/2)+1), np.arange(-round(n1/2)+1, math.floor(n1/2)+1))
    d = np.sqrt(k1**2 + k2**2)
    
    # reject filter at frequencies between freq1 and freq2
    reject = np.ones_like(d)
    reject[(d > freq1) & (d < freq2)] = 0

    h = reject

    if visualize:
        visualize_filter(h)

    return h

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

# apply filter
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
img_corrupt = add_periodic_noise(img, amplitude=25, frequency=2)

""" Magnitude spectra """
# noisy image
F_25 = np.fft.fft2(img_corrupt)
fshift2 = np.fft.fftshift(F_25)
magnitude_spectrum_corrupt = np.abs(fshift2)
magnitude_spectrum_corrupt = np.log(magnitude_spectrum_corrupt + 1)

# filtered image and magnitude spectra
radius=3; n=8
H = H_NR(*np.meshgrid(np.arange(F_25.shape[0]), np.arange(F_25.shape[1]), indexing='ij'), d0=radius, n=n)

f1 = np.fft.fftshift(np.fft.fft2(img_corrupt))
filtered = H * f1

fshift = np.fft.fftshift(filtered)
magnitude_spectrum_filtered = np.abs(fshift)  
magnitude_spectrum_filtered = np.log(magnitude_spectrum_filtered + 1) 

filtered_img = np.fft.ifft2(np.fft.ifftshift(filtered)).real

""" 1D Slices """
# noisy
middle_row_corrupt = magnitude_spectrum_corrupt[magnitude_spectrum_corrupt.shape[0]//2,:]
middle_column_corrupt = magnitude_spectrum_corrupt[:,magnitude_spectrum_corrupt.shape[1]//2]

# filtered
middle_row_filtered = magnitude_spectrum_filtered[magnitude_spectrum_filtered.shape[0]//2,:]
middle_col_filtered = magnitude_spectrum_filtered[:,magnitude_spectrum_filtered.shape[1]//2]

""" Plots """
# filtered 1D slices
plt.plot(middle_row_filtered)
plt.title('Middle Row 1D Slice Filtered')
plt.ylabel('Magnitude')
plt.show()

plt.plot(middle_col_filtered)
plt.title('Middle Column 1D Slice Filtered')
plt.ylabel('Magnitude')
plt.show()

""" Filtered Image"""
plt.imshow(filtered_img, cmap='gray')
plt.title('Filtered Image')
plt.show()

# plot magnitude spectrum filtered
plt.imshow(magnitude_spectrum_filtered)
plt.title('Magnitude Spectrum Filtered')
plt.axis('off')
plt.colorbar()
plt.show()

""" Display Filter """
# display filter 2D FFT
plt.imshow(H, cmap='gray')
plt.title('Filter')
plt.show()

f = np.fft.fft2(H)
fshift = np.fft.fftshift(f)

# Compute the magnitude spectrum
magnitude_spectrum = np.abs(fshift)
magnitude_spectrum = np.log(magnitude_spectrum + 1)

# Display the magnitude spectrum
plt.imshow(magnitude_spectrum)
plt.title('2D FFT Butterworth Bandreject Notchfilter')
plt.show()

# Compute the middle row and column of the magnitude spectrum
middle_row = magnitude_spectrum[magnitude_spectrum.shape[0]//2,:]
middle_col = magnitude_spectrum[:,magnitude_spectrum.shape[1]//2]

# 1D slices 
plt.plot(middle_row)
plt.title('Middle Row Magntidue Notch Reject')
plt.show()

plt.plot(middle_col)
plt.title('Middle Column Magntidue Notch Reject')
plt.show()

