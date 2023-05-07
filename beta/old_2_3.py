import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from scipy.ndimage import gaussian_filter

# TODO:
#   gaussian filter
#   wiener filter
#       >> explain the sensitivity to noise of the edge detection kernel


def add_gaussian_noise(img, mean, stdev):

        # build noise mask with equal dimensions as the input image
        noise_mask = np.random.normal(mean, stdev, size=img.shape)
    
        # add the noise mask
        noisy_image = np.add(img, noise_mask) 
        
        # clip
        noisy_image = np.clip(noisy_image, 0, 255)

        # convert back 
        noisy_image = noisy_image.astype(np.uint8)

        return noisy_image
        
def wiener_filter(img, kernel, K):
    # Normalize the kernel sum
    kernel /= np.sum(kernel)

    # Make a copy of the image
    dummy = np.copy(img)

    # Apply FFT to the image and kernel
    dummy = fft2(dummy)
    kernel = fft2(kernel, s=img.shape)

    # Calculate the Wiener filter coefficients
    kernel = np.conj(kernel) / (np.abs(kernel)**2 + K)

    # Apply the Wiener filter to the image
    dummy = dummy * kernel

    # Apply inverse FFT and take the absolute value
    dummy = np.abs(ifft2(dummy))

    return dummy

def gaussian_kernel(kernel_size=3):
    # Generate a 1D Gaussian kernel
    kernel_1d = gaussian_filter(np.ones((kernel_size,)), kernel_size / 3)

    # Create a 2D Gaussian kernel by outer product
    kernel_2d = np.outer(kernel_1d, kernel_1d)

    # Normalize the kernel
    kernel_2d /= np.sum(kernel_2d)

    return kernel_2d

# adapted from C++ code on this link: 
#   https://answers.opencv.org/question/36309/opencv-gaussian-noise/
def gaussian_noise(imGray):
    noise = np.zeros_like(imGray, dtype=np.float64)
    cv.normalize(imGray, result, 0.0, 1.0, cv.NORM_MINMAX, dtype=cv.CV_64F)
    cv.randn(noise, 0, 0.05)
    result = result + noise
    return cv.normalize(result, result, 0.0, 1.0, cv.NORM_MINMAX, dtype=cv.CV_64F)

    

# fastnldmean

# this one is taken from the labs
diag_45 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])

# clockwise rotate it 90 degree
diag_135 = np.rot90(diag_45, 1)

# load iamge
img = cv.imread('images/diag2.jpg', cv.COLOR_BGR2GRAY) 

# add gaussian noise
img_noisy = add_gaussian_noise(img, 0, 10)

# apply edge detection 45-deg kernel to noisy image
img_noisy_diag_45 = cv.filter2D(src=img_noisy,ddepth=-1,kernel=diag_45)

# apply wiener filter
# kernel = gaussian_kernel()
# img_wiener = wiener_filter(img_noisy,kernel,10)

# img cleaned
img_cleaned = cv.fastNlMeansDenoising(img_noisy_diag_45)

# apply edge-detection
img_wiener_prewitt45 = cv.filter2D(src=img_cleaned,ddepth=-1,kernel=diag_45)

# Plot original image
plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Plot noisy image
plt.subplot(2, 3, 2)
plt.imshow(img_noisy)
plt.title('Image with Gaussian Noise')
plt.axis('off')

# Plot noisy image with edge detection kernel 45-deg applied
plt.subplot(2, 3, 3)
plt.imshow(img_noisy_diag_45)
plt.title('Noisy Image Edge 45-deg')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(img_cleaned)
plt.title('Cleaned Filter Image')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(img_wiener_prewitt45)
plt.title('Wiener Filter Edge detection')
plt.axis('off')

# Show results
plt.tight_layout()
plt.show()

