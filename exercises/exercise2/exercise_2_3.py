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
img_noisy = add_gaussian_noise(img,0,0.1)

# apply edge detection 45-deg kernel to noisy images
img_noisy_45 = cv.filter2D(src=img_noisy,ddepth=-1,kernel=diag_45)
img_noisy_135 = cv.filter2D(src=img_noisy,ddepth=-1,kernel=diag_135)

# denoise the image
img_denoise = cv.GaussianBlur(img_noisy,(5,5), 0)

# apply edge detection to denoised images
img_denoise_45_edge = cv.filter2D(src=img_denoise,ddepth=-1,kernel=diag_45)
img_denoise_135_edge = cv.filter2D(src=img_denoise,ddepth=-1,kernel=diag_135)

# TODO: (make one plot with 6 subplots...)
#   plot original and noisy image
#   plot edge detection images on noisy images
#   plot edge detection images on denoised images 

# create 6-subplot figure
fig, axs = plt.subplots(2,2, figsize=(12, 8))



# plot edge detection on noisy image
axs[0, 0].imshow(img_noisy_45, cmap='gray')
axs[0, 0].set_title('Noisy (45)')

# plot denoised image
axs[0, 1].imshow(img_noisy_135, cmap='gray')
axs[0, 1].set_title('Noisy (135)')

# plot edge detection on denoised image
axs[1, 0].imshow(img_denoise_45_edge, cmap='gray')
axs[1, 0].set_title('Denoised (45)')

# plot median denoised image
axs[1, 1].imshow(img_denoise_135_edge, cmap='gray')
axs[1, 1].set_title('Denoised (135)')

# set global title
fig.suptitle('Image Denoising and Edge Detection')

# show the figure
plt.show()