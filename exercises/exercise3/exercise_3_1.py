import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# method to build 2D adapted from:
#   https://thepythoncodingbook.com/2022/05/28/numpy-meshgrid/
def build_2D_sine(img, amplitude, frequency):

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

# load image
img = cv.imread('images/birdie.jpg', cv.COLOR_BGR2GRAY)

# add periodic noise
X,Y, sine2D = build_2D_sine(img,amplitude=10, frequency=2)

# Plot the sine pattern
plt.imshow(sine2D)
plt.title('Meshgrid Sinusoid 2D Projection')
plt.colorbar()
plt.axis('off')
plt.show()

# Plot the meshgrid in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, sine2D)
ax.set_title('Meshgrid Sinusoid 3D Projection')
plt.show()

# Extra 
f = np.fft.fft2(sine2D)
fshift = np.fft.fftshift(f)
magnitude_spectrum = np.abs(fshift)
magnitude_spectrum = np.log(magnitude_spectrum + 1)
plt.imshow(magnitude_spectrum)
plt.title('Magnitude Spectrum Meshgrid Sinsuid 2D')
plt.colorbar()
plt.axis('off')
plt.show()

# POINTS:
#   (1417,1206) & (1417,1199)
#   for freq=10, amp=10
