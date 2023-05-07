import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# method to build meshgrid 2D sine based on image dimensions
def build_2D_sine(img, amplitude=1.5, frequency=33):

    # grayscale
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    dimensions = img.shape

    print('dimensions', dimensions)

    # build meshgrid based on image dimensions
    x = np.linspace(-1,1,dimensions[1])
    y = np.linspace(-1,1,dimensions[0])

    X, Y = np.meshgrid(x,y)

    # build the 2D sine, adjusting X and Y in output, controls orientation
    #   f(x,y) = sin(2*pi*x/lambda)
    
    wavelength = 1/frequency

    sine2D = amplitude * np.sin(2*np.pi*Y/wavelength)

    return X, Y, sine2D

    # now add the noise

# load image
img = cv.imread('images/birdie.jpg', cv.COLOR_BGR2GRAY)

# add periodic noise
X,Y, sine2D = build_2D_sine(img,1.0,2)

# Plot the sine pattern
plt.imshow(sine2D, cmap='gray')
plt.title('2D Sine Pattern')
plt.colorbar()
plt.axis('off')
plt.show()

# Plot the meshgrid in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, sine2D, cmap='gray')
ax.set_title('Meshgrid 3D Sine Pattern')
plt.show()
