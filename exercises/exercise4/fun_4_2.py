import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

# read in the image
img = cv2.imread('images/colors.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# get the shape of the image
rows, cols, _ = img_rgb.shape

# define the transformation function
def polar_transform(point):
    x, y = point
    rho = math.sqrt(x**2 + y**2)
    phi = math.atan2(y, x)
    rho_prime = math.sqrt(rho)
    return (int(rho_prime * math.cos(phi)), int(rho_prime * math.sin(phi)))

# apply the transformation using inverse mapping
transformed_img = np.zeros_like(img_rgb)
for x in range(cols):
    for y in range(rows):
        x_prime, y_prime = polar_transform((x, y))
        if x_prime < cols and y_prime < rows:
            transformed_img[y, x] = img_rgb[y_prime, x_prime]

# display the original and transformed images side-by-side
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img_rgb)
ax[0].set_title('Original Image')
ax[1].imshow(transformed_img)
ax[1].set_title('Transformed Image')
plt.show()