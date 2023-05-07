import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
im_gray = cv2.imread('images/birdie.jpg', cv2.IMREAD_GRAYSCALE)

# Create noise matrix
noise = np.zeros_like(im_gray, dtype=np.float64)

# Normalize the grayscale image to a range of [0, 1]
result = cv2.normalize(im_gray.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

# Generate noise with a mean of 0 and standard deviation of 0.05
cv2.randn(noise, 0, 0.2)

# Add noise to the result
result = result + noise

# Normalize the result to a range of [0, 1]
result = cv2.normalize(result, None, 0.0, 1.0, cv2.NORM_MINMAX)

# plot
plt.subplot(1, 2, 1)
plt.imshow(im_gray, cmap='gray')
plt.title('Clean Image')

plt.subplot(1, 2, 2)
plt.imshow(result, cmap='gray')
plt.title('Image with Added Gaussian Noise')

plt.show()
