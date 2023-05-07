import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2

# adapted from C++ code on this link: 
#   https://answers.opencv.org/question/36309/opencv-gaussian-noise/
def add_gaussian_noise(img,mean,std_dev):

    # noise matrix
    noise = np.zeros_like(img, dtype=np.float64)

    # normalize the grayscale image to a range of [0, 1]
    result = cv.normalize(img.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

    # generate gaussian noise
    cv.randn(noise, mean, std_dev)

    # add noise to the result
    result = result + noise

    # normalize the result to a range of [0, 1]
    result = cv.normalize(result, None, 0.0, 1.0, cv.NORM_MINMAX)

    return result

# from notes made during lecture
diag_135 = np.array([[0,1,1],[-1,0,1],[-1,-1,0]])

# clockwise rotate it 90 degree
diag_45 = np.rot90(diag_135, 1)

# load image
img = cv.imread('images/diag2.jpg', cv.COLOR_BGR2GRAY) 

# add varying levels of gaussian noise
img_noisy = add_gaussian_noise(img,0,0.05)
img_noisy_high = add_gaussian_noise(img,0,0.2)

# apply edge detection 45-deg kernel to noisy images
img_noisy_45_low = cv.filter2D(src=img_noisy,ddepth=-1,kernel=diag_45)
img_noisy_135_low = cv.filter2D(src=img_noisy,ddepth=-1,kernel=diag_135)

img_noisy_45_high = cv.filter2D(src=img_noisy_high,ddepth=-1,kernel=diag_45)
img_noisy_135_high = cv.filter2D(src=img_noisy_high,ddepth=-1,kernel=diag_135)

# denoise the image
img_denoise_low = cv.GaussianBlur(img_noisy, (15,15), 1.4)
img_denoise_high = cv.GaussianBlur(img_noisy_high, (15,15), 1.4)

# apply edge detection to denoised image 
img_denoise_45_low = cv.filter2D(src=img_denoise_low,ddepth=-1,kernel=diag_45)
img_denoise_135_low = cv.filter2D(src=img_denoise_high,ddepth=-1,kernel=diag_135)

# Plot original image
plt.subplot(1, 2, 1)
plt.imshow(img_noisy, cmap='gray')
plt.title('Low Noise Level Image')
plt.axis('off')

# Plot noisy image
plt.subplot(1, 2, 2)
plt.imshow(img_noisy_high)
plt.title('High Noise Level Image')
plt.axis('off')

plt.tight_layout()
plt.show()

# Plot noisy image with edge detection kernel 45-deg applied
# plt.subplot(2, 2, 1)
# plt.imshow(img_noisy_45_low)
# plt.title('Low Noise Level 45-deg')
# plt.axis('off')

# plt.subplot(2, 2, 2)
# plt.imshow(img_noisy_135_low)
# plt.title('Low Noise Level 135-deg')
# plt.axis('off')

# plt.subplot(2, 2, 3)
# plt.imshow(img_noisy_45_high)
# plt.title('High Noise Level 45-deg')
# plt.axis('off')


# plt.subplot(2, 2, 4)
# plt.imshow(img_noisy_135_high)
# plt.title('High Noise Level 135-deg')
# plt.axis('off')

# plt.tight_layout()
# plt.show()

# denoised image
plt.subplot(1, 2, 1)
plt.imshow(img_denoise_low)
plt.title('Low Level Denoised Image using Median Blur')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_denoise_high)
plt.title('High Level Denoised Image using Median Blur')
plt.axis('off')

plt.tight_layout()
plt.show()

# again edge-detection...
plt.subplot(2, 2, 1)
plt.imshow(img_denoise_45_low)
plt.title('Low Level Denoise 45-deg')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(img_denoise_135_low)
plt.title('Low Level Denoise 135-deg')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(img_noisy_45_high)
plt.title('High Level Denoise 45-deg')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(img_noisy_135_high)
plt.title('High Level Denoise 135-deg')
plt.axis('off')

plt.tight_layout()
plt.show()
