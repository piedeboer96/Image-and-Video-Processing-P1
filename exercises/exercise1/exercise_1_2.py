import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# load images BGR 
img_1 = cv.imread('images/flamingo.jpg')       
img_2 = cv.imread('images/deer.jpg')  

# convert to HSV
img_1_hsv = cv.cvtColor(img_1, cv.COLOR_BGR2HSV)
img_2_hsv = cv.cvtColor(img_2, cv.COLOR_BGR2HSV)

# split channels HSV 
h1, s1, v1 =  cv.split(img_1_hsv)
h2, s2, v2 =  cv.split(img_2_hsv)

# histogram equalization for seperate H,S,V channels
h1_eq = cv.equalizeHist(h1); s1_eq = cv.equalizeHist(s1); v1_eq = cv.equalizeHist(v1)
h2_eq = cv.equalizeHist(h2); s2_eq = cv.equalizeHist(s2); v2_eq = cv.equalizeHist(v2)

"""
Plotting of histograms for image 1
"""

plt.figure(figsize=(12,10))

# hue
plt.subplot(2,3,1)
plt.hist(h1.flatten(), bins=256, range=[0, 256], alpha=0.5)
plt.title('Histogram Hue (Image 1)')

plt.subplot(2,3,4)
plt.hist(h1_eq.flatten(), bins=256, range=[0, 256], alpha=0.5)
plt.title('Histogram EQ Hue (Image 1)')

# saturation
plt.subplot(2,3,2)
plt.hist(s1.flatten(), bins=256, range=[0, 256], alpha=0.5)
plt.title('Histogram Saturation (Image 1)')

plt.subplot(2,3,5)
plt.hist(s1_eq.flatten(), bins=256, range=[0, 256], alpha=0.5)
plt.title('Histogram EQ Saturation (Image 1)')

# value
plt.subplot(2,3,3)
plt.hist(v1.flatten(), bins=256, range=[0, 256], alpha=0.5)
plt.title('Histogram Value (Image 1)')

plt.subplot(2,3,6)
plt.hist(v1_eq.flatten(), bins=256, range=[0, 256], alpha=0.5)
plt.title('Histogram EQ Value (Image 1)')

plt.tight_layout()
plt.show()

"""
Display original and equalized image 2
"""

# build image with equalized H,S,V channels by merging the channels
img_1_eq_hsv = cv.merge((h1_eq, s1_eq, v1_eq))
img_2_eq_hsv = cv.merge((h2_eq, s2_eq, v2_eq))

# convert to BGR
img_1_eq_bgr = cv.cvtColor(img_1_eq_hsv, cv.COLOR_HSV2BGR)
img_2_eq_bgr = cv.cvtColor(img_2_eq_hsv, cv.COLOR_HSV2BGR)

# plot original and equalized images for image 1
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img_1, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Original Color Image')
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(img_1_eq_hsv, cv.COLOR_HSV2RGB))
plt.axis('off')
plt.title('Equalized Color Image')
plt.tight_layout()
plt.show()

"""
Plotting of histograms for image 2
"""

plt.figure(figsize=(12,10))

# hue
plt.subplot(2,3,1)
plt.hist(h2.flatten(), bins=256, range=[0, 256], alpha=0.5)
plt.title('Histogram Hue (Image 2)')

plt.subplot(2,3,4)
plt.hist(h2_eq.flatten(), bins=256, range=[0, 256], alpha=0.5)
plt.title('Histogram EQ Hue (Image 2)')

# saturation
plt.subplot(2,3,2)
plt.hist(s2.flatten(), bins=256, range=[0, 256], alpha=0.5)
plt.title('Histogram Saturation (Image 2)')

plt.subplot(2,3,5)
plt.hist(s2_eq.flatten(), bins=256, range=[0, 256], alpha=0.5)
plt.title('Histogram EQ Saturation (Image 2)')

# value
plt.subplot(2,3,3)
plt.hist(v2.flatten(), bins=256, range=[0, 256], alpha=0.5)
plt.title('Histogram Value (Image 2)')

plt.subplot(2,3,6)
plt.hist(v2_eq.flatten(), bins=256, range=[0, 256], alpha=0.5)
plt.title('Histogram EQ Value (Image 2)')

plt.tight_layout()
plt.show()

"""
Display original and equalized image 2
"""

# plot original and equalized images for image 1
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img_2, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Original Color Image')
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(img_2_eq_hsv, cv.COLOR_HSV2RGB))
plt.title('Equalized Color Image')
plt.axis('off')
plt.tight_layout()
plt.show()