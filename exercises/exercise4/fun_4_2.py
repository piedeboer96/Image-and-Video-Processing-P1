import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('images/palette.jpg')

# Convert the image to polar coordinates
height, width = image.shape[:2]
center = (width // 2, height // 2)
max_radius = int(np.sqrt(center[0] ** 2 + center[1] ** 2))

# Perform the polar coordinate transformation
polar_image = cv2.warpPolar(image, (width, height), center, max_radius, cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

# Display the original and transformed images
fig, ax = plt.subplots(1, 2)
ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax[0].set_title('Original Image')
ax[0].axis('off')
ax[1].imshow(cv2.cvtColor(polar_image, cv2.COLOR_BGR2RGB))
ax[1].set_title('Transformed Image (Polar Coordinates)')
ax[1].axis('off')
plt.show()