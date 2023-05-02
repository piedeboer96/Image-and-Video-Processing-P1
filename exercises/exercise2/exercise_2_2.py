import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

"""
        - Salt and pepper method
        - Median Filter
        - Kernels 45-deg and 135-deg from lab
        - Median Filter
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

# diagonal 45
diag_45 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])

# clockwise rotate it 90 degree
diag_135 = np.rot90(diag_45, 1)

##############################

# load iamge
img = cv.imread('images/diag2.jpg', cv.COLOR_BGR2GRAY) 

# add salt and pepper
img_noisy = add_salt_and_pepper(img, 0.5)

# apply the different kernels using filter2D
img_noisy_45 = cv.filter2D(src=img_noisy,ddepth=-1,kernel=diag_45)
img_noisy_135 = cv.filter2D(src=img_noisy, ddepth=-1,kernel=diag_135)

# plot original image
plt.subplot(2, 2, 1); plt.imshow(img); plt.title('Original Image'); plt.axis('off')

# plot noisy image
plt.subplot(2, 2, 2); plt.imshow(img_noisy); plt.title('Image with Salt and Pepper'); plt.axis('off')

# plot noisy image with edge detection kernel 45-deg applied
plt.subplot(2, 2, 3); plt.imshow(img_noisy_45); plt.title('Noisy Image Edge Detection 45-deg'); plt.axis('off')

# plot noisy image with 
plt.subplot(2, 2, 4); plt.imshow(img_noisy_135); plt.title('Noisy Image Edge Detection 135-deg'); plt.axis('off')

# show results
plt.tight_layout()
plt.show()

# apply median filter to noisy image with kernel size 5 
img_filter = cv.medianBlur(img_noisy, 5)

# apply the different edge detection kernels to the median filtered noisy images
img_filter_45 = cv.filter2D(src=img_filter,ddepth=-1,kernel=diag_45)
img_filter_135 = cv.filter2D(src=img_filter,ddepth=-1,kernel=diag_135)

# plot the result
plt.subplot(2, 2, 1); plt.imshow(img_noisy); plt.title('Noisy Image'); plt.axis('off')

# plot noisy image
plt.subplot(2, 2, 2); plt.imshow(img_filter); plt.title('Filtered Image'); plt.axis('off')

# plot noisy image with edge detection kernel 45-deg applied
plt.subplot(2, 2, 3); plt.imshow(img_filter_45); plt.title('Filtered Image with Edge Detection 45-deg'); plt.axis('off')

# plot noisy image with 
plt.subplot(2, 2, 4); plt.imshow(img_filter_135); plt.title('Filtered Image with Edge Detection 135-deg'); plt.axis('off')

# show results
plt.tight_layout()
plt.show()