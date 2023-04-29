import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# load image
img = cv.imread('images/birdie.jpg', cv.IMREAD_GRAYSCALE)

""" code for 2d sine meshgrid """

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
output = amplitude * np.sin(2*np.pi*Y/wavelength)

# compute 2d fft 
np.fft