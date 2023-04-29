import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# load image
img = cv.imread('images/birdie.jpg', cv.COLOR_BGR2GRAY)

# get dimensions of image
dimensions = img.shape

print('dimensions', dimensions)

# build meshgrid based on image dimensions
x = np.linspace(-1,1,dimensions[0])
y = np.linspace(-1,1,dimensions[1])

X, Y = np.meshgrid(x,y)

# build the 2D sine, adjusting X and Y in output, controls orientation
#   f(x,y) = sin(2*pi*x/lambda)
wavelength = 0.5  
amplitude = 1.0 
output = amplitidue * np.sin(2*np.pi*Y/wavelength)

# plot
#plt.imshow(output)
#plt.show()

# Plot the sine pattern
plt.imshow(output, cmap='gray')
plt.title('2D Sine Pattern')
plt.colorbar()
plt.axis('off')
plt.show()

# Plot the meshgrid in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, output, cmap='gray')
ax.set_title('Meshgrid in 3D')
plt.show()



# (code from the guy...)
fig = plt.figure()
ax = fig.add_subplot(121)
ax.imshow(
    output,
    cmap="copper",
    extent=[np.min(x), np.max(x), np.min(y), np.max(y)],
)
ax = fig.add_subplot(122, projection="3d")
ax.plot_surface(X, Y, output, cmap="copper")
plt.show()

