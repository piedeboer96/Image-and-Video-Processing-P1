import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

"""
    Methods
"""

# method to add salt and pepper
def add_salt_and_pepper(img, prob):
    # copy the image to build the noisy image
    image_noisy = np.copy(img)

    # generate random noise mask with the same shape as the image
    noise_mask = np.random.random(image_noisy.shape)
    
    # set salt (white pixels) where noise mask is less than prob/2
    image_noisy[noise_mask < prob / 2] = 255

    # set pepper (black pixels) where noise mask is greater than 1 - prob/2
    image_noisy[noise_mask > 1 - prob / 2] = 0

    return image_noisy

# from notes made during lecture
diag_135 = np.array([[0,1,1],[-1,0,1],[-1,-1,0]])

# clockwise rotate it 90 degree
diag_45 = np.rot90(diag_135, 1)

##############################

# load iamge
img = cv.imread('images/diag2.jpg', cv.COLOR_BGR2GRAY) 

# add salt and pepper
img_noisy = add_salt_and_pepper(img, 0.5)

# apply the different kernels using filter2D
img_noisy_45 = cv.filter2D(src=img_noisy,ddepth=-1,kernel=diag_45)
img_noisy_135 = cv.filter2D(src=img_noisy, ddepth=-1,kernel=diag_135)

(T, binary_45) = cv.threshold(img_noisy_45, 100, 255, cv.THRESH_BINARY)
(T, binary_135) = cv.threshold(img_noisy_135, 100, 255, cv.THRESH_BINARY)

# plot original image
plt.subplot(2, 2, 1); plt.imshow(img); plt.title('Original Image'); plt.axis('off')

# plot noisy image
plt.subplot(2, 2, 2); plt.imshow(img_noisy); plt.title('Image with Impulse Noise'); plt.axis('off')

# plot noisy image with edge detection kernel 45-deg applied
plt.subplot(2, 2, 3); plt.imshow(img_noisy_45); plt.title('Noisy Image Edge Detection 45-deg'); plt.axis('off')

# plot noisy image with 
plt.subplot(2, 2, 4); plt.imshow(img_noisy_135); plt.title('Noisy Image Edge Detection 135-deg'); plt.axis('off')

# show results
plt.tight_layout()
plt.show()


# binarized results...
plt.subplot(2, 2, 1); plt.imshow(binary_45); plt.title('Noisy Image Edge Detection 45-deg Binary'); plt.axis('off')

# plot noisy image
plt.subplot(2, 2, 2); plt.imshow(binary_135); plt.title('Noisy Image Edge Detection 135-deg Binary'); plt.axis('off')

# show results
plt.tight_layout()
plt.show()


# apply median filter to noisy image with kernel size 5 
img_filter = cv.medianBlur(img_noisy, 5)

# apply the different edge detection kernels to the median filtered noisy images
img_filter_45 = cv.filter2D(src=img_filter,ddepth=-1,kernel=diag_45)
img_filter_135 = cv.filter2D(src=img_filter,ddepth=-1,kernel=diag_135)

# binarize the filter image result
(T, binary_filter_45) = cv.threshold(img_filter_45, 100, 255, cv.THRESH_BINARY)
(T, binary_filter_135) = cv.threshold(img_filter_135, 100, 255, cv.THRESH_BINARY)

# noisy image
plt.subplot(2, 2, 1); plt.imshow(img_noisy); plt.title('Image with Impulse Noise'); plt.axis('off')

# filtered image
plt.subplot(2, 2, 2); plt.imshow(img_filter); plt.title('Filtered Image'); plt.axis('off')

# plot noisy image with edge detection kernel 45-deg applied
plt.subplot(2, 2, 3); plt.imshow(binary_filter_45); plt.title('Filtered Image with Edge Detection 45-deg'); plt.axis('off')

# plot noisy image with edge detection kernel 45-deg applied + binarization
plt.subplot(2, 2, 4); plt.imshow(binary_filter_135); plt.title('Filtered Image with Edge Detection 135-deg'); plt.axis('off')

# show results
plt.tight_layout()
plt.show()