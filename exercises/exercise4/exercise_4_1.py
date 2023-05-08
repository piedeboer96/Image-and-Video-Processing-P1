import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


# book 10.5 - clustering is used for image/region segmentation

# k-means code from the labs
def k_means(image, k):
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    return segmented_image

# soble kernel taken from fig. 3.50 from the book (Gonzel)
sobel_kernel = np.array([[-1, -2, -1],[0,0,0],[1,2,1]])

# using the follow approach we can obtain a cartoony image
#   1. apply edge detection (sobel or canny) to first copy 
#   2. take the 'negative' of this edge image
#   2. apply clustering (k-means) to the second copy
#   3. overlay the images (= cartoonified)

img = cv2.imread('images/sheep.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# k=3, inspired from fig (10.49)
img_k_means = k_means(img_rgb, 3)

# apply soble using openCV filter2D
img_sobel = cv2.filter2D(src=img,ddepth=-1,kernel=sobel_kernel)

# treshold and take the negative for the 'black edges' (= grayscale set complementation)
threshold_value = 100
ret, img_sobel_thresh = cv2.threshold(img_sobel, threshold_value, 255, cv2.THRESH_BINARY)
img_sobel_neg = cv2.bitwise_not(img_sobel_thresh)

# overlay the images to create the cartoon effect
img_cartoon = np.bitwise_and(img_sobel_neg, img_k_means)

# increase saturation by clipping 'saturation' channel
img_cartoon_hsv = cv2.cvtColor(img_cartoon, cv2.COLOR_RGB2HSV)
img_cartoon_hsv[:,:,1] = np.clip(img_cartoon_hsv[:,:,1] * 1.5, 0, 255)
img_cartoon = cv2.cvtColor(img_cartoon_hsv, cv2.COLOR_HSV2RGB)

# display all 3 images
plt.subplot(221), plt.imshow(img_sobel), plt.title('Sobel Filter')
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(img_sobel_neg), plt.title('Sobel Negative (Black Edge)')
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(img_k_means), plt.title('K-Means Clustering')
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(img_cartoon), plt.title('Cartoonified Image (Saturated)')
plt.xticks([]), plt.yticks([])
plt.show()