import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# load images
img_1 = cv.imread('images/flamingo.jpg')
img_2 = cv.imread('images/deer.jpg')

# convert to HSV 
img_1_hsv = cv.cvtColor(img_1, cv.COLOR_BGR2HSV)
img_2_hsv = cv.cvtColor(img_2, cv.COLOR_BGR2HSV)

# split channels
h1, s1, v1 = cv.split(img_1_hsv)
h2, s2, v2 = cv.split(img_2_hsv)

# apply histogram equalization only to the V channel
v1_eq = cv.equalizeHist(v1)
v2_eq = cv.equalizeHist(v2)

# build image with equalized V channel by merging the appropriate channels
img_1_eq_hsv = cv.merge((h1, s1, v1_eq))
img_2_eq_hsv = cv.merge((h2, s2, v2_eq))

# conver to BGR
img_1_eq_bgr = cv.cvtColor(img_1_eq_hsv, cv.COLOR_HSV2BGR)
img_2_eq_bgr = cv.cvtColor(img_2_eq_hsv, cv.COLOR_HSV2BGR)

# plot histograms (image 1)
# plt.figure(figsize=(12, 8))
# plt.subplot(2, 2, 1)
# plt.hist(v1.flatten(), bins=256, range=[0, 256], color='b', alpha=0.5)
# plt.title('Original Value Histogram')
# plt.subplot(2, 2, 2)
# plt.hist(v1_eq.flatten(), bins=256, range=[0, 256], color='b', alpha=0.5)
# plt.title('Equalized Value Histogram')

# plot original and equalized images
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img_1, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Original Color Image')
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(img_1_eq_bgr, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Equalized Color Image')
plt.tight_layout()
plt.show()

# plot histograms (image 2)
# plt.figure(figsize=(12, 8))
# plt.subplot(2, 2, 1)
# plt.hist(v2.flatten(), bins=256, range=[0, 256], color='b', alpha=0.5)
# plt.title('Original Value Histogram')
# plt.subplot(2, 2, 2)
# plt.hist(v2_eq.flatten(), bins=256, range=[0, 256], color='b', alpha=0.5)
# plt.title('Equalized Value Histogram')

# Plot the original and equalized color images
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img_2, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Original Color Image')
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(img_2_eq_bgr, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Equalized Color Image')
plt.tight_layout()
plt.show()