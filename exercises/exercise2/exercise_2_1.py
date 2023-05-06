import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# load image
img = cv.imread('images/diag2.jpg', cv.COLOR_BGR2GRAY) 

# from notes made during lecture
diag_135 = np.array([[0,1,1],[-1,0,1],[-1,-1,0]])

# clockwise rotate it 90 degree
diag_45 = np.rot90(diag_135, 1)

# apply the different kernels using filter2D
img_45 = cv.filter2D(src=img,ddepth=-1,kernel=diag_45)
img_135 = cv.filter2D(src=img, ddepth=-1,kernel=diag_135)

# apply binary tresholding
(T, binary_45) = cv.threshold(img_45, 100, 255, cv.THRESH_BINARY)
(T, binary_135) = cv.threshold(img_135, 100, 255, cv.THRESH_BINARY)

# create a matplotlib figure with 2x2 subplots
plt.figure(figsize=(12, 10))

# plot original image
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

# plot the 45-degrees edge-detection + tresholding image
plt.subplot(2, 2, 2)
plt.imshow(binary_45, cmap='gray')
plt.title('45-degree 1st Order Spatial Edge Detection with Binary Treshold')
plt.axis('off')

# plot original image  
plt.subplot(2, 2, 3)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

# Plot the image after applying the 135-degree edge detection kernel
plt.subplot(2, 2, 4)
plt.imshow(binary_135, cmap='gray')
plt.title('135-degree 1st Order Spatial Edge Detection with Binary Treshold')
plt.axis('off')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the figure
plt.show()
