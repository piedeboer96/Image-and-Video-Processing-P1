import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

""" 
    Background 
"""

# GEOMETRIC TRANSFOMRATION (page 102)
# INTERPOLATION (page 77)

#    1. Spatial tranfromation of coordinates
#    2. Intestity interpolation that assigns intenstity 
#       values to the spatially transformed pixels

# coordinate equations ,

#   phi' = phi
#   rho' = sqrt(rho)

"""
    Methods
"""

# capture the coordinate equations.
def polar_coordinate_transformation(x,y):
    rho = math.sqrt(x**2 + y**2)
    phi = math.atan2(y, x)
    rho_prime = math.sqrt(rho)
    return (rho_prime * math.cos(phi)), (rho_prime * math.sin(phi))  # fix here, swapped x and y


def inverse_nearest_neighbour(img_rgb):
    
    rows, cols, channels = img_rgb.shape

    # initialize transformed image
    img_transformed = np.zeros_like(img_rgb)

    for c in range(channels):
        for i in range(rows):
            for j in range(cols):

                # apply polar coordinate transformation
                x, y = polar_coordinate_transformation(j-cols/2, i-rows/2)

                # apply inverse nearest neighbour interpolation
                new_x = int(round(x + cols/2))
                new_y = int(round(y + rows/2))

                if 0 <= new_x < cols and 0 <= new_y < rows:
                    img_transformed[new_y, new_x, c] = img_rgb[i, j, c]
    
    return img_transformed

"""
    Demonstration
"""

# load image
img = cv2.imread('images/palette.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# run the nearest neigbour 'inverse' version
img_transformed = inverse_nearest_neighbour(img_rgb)
img_transformed = cv2.resize(img_transformed, (img_rgb.shape[1],img_rgb.shape[0]))

print(img_transformed[0])

# display original and transformed image
plt.subplot(221), plt.imshow(img_rgb), plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(img_transformed), plt.title('Transformed Image')
plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
