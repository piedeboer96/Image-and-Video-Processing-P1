import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# load images BGR 
img_1 = cv.imread('images/flamingo.jpg')       
img_2 = cv.imread('images/deer.jpg')           

# convert to HSV
img_1_hsv = cv.cvtColor(img_1, cv.COLOR_BGR2HSV)
img_2_hsv = cv.cvtColor(img_2, cv.COLOR_BGR2HSV)

# split channels RGB and HSV image 1
b1, g1, r1 =  cv.split(img_1)
h1, s1, v1 =  cv.split(img_1_hsv)

# split channels RGB and HSV image 2
b2, g2, r2 =  cv.split(img_2)
h2, s2, v2 =  cv.split(img_2_hsv)

# build a plot for image 1 using appropriate color maps
fig = plt.figure(figsize=(10, 7));rows=2;columns=3

# RGB image 1
fig.add_subplot(rows, columns, 1); plt.imshow(r1, cmap='Reds'); plt.axis('off'); plt.title("Red Channel Image 1")
fig.add_subplot(rows, columns, 2); plt.imshow(g1, cmap='Greens'); plt.axis('off'); plt.title("Green Channel Image 1")
fig.add_subplot(rows, columns, 3); plt.imshow(b1, cmap='Blues'); plt.axis('off'); plt.title("Blue Channel Image 1")

# HSV image 1
fig.add_subplot(rows, columns, 4); plt.imshow(h1, cmap='gray'); plt.axis('off'); plt.title("Hue Channel Image 1")
fig.add_subplot(rows, columns, 5); plt.imshow(s1, cmap='gray'); plt.axis('off'); plt.title("Saturation Channel Image 1")
fig.add_subplot(rows, columns, 6); plt.imshow(v1, cmap='gray'); plt.axis('off'); plt.title("Value Channel Image 1")

# show plot 
plt.tight_layout()
plt.show()

# build a plot for image 2 using appropriate color maps
fig = plt.figure(figsize=(10, 7));rows=2;columns=3

# RGB image 2
fig.add_subplot(rows, columns, 1); plt.imshow(r2, cmap='Reds'); plt.axis('off'); plt.title("Red Channel (Image 2)")
fig.add_subplot(rows, columns, 2); plt.imshow(g2, cmap='Greens'); plt.axis('off'); plt.title("Green Channel (Image 2)")
fig.add_subplot(rows, columns, 3); plt.imshow(b2, cmap='Blues'); plt.axis('off'); plt.title("Blue Channel (Image 2)")

# HSV image 2
fig.add_subplot(rows, columns, 4); plt.imshow(h2, cmap='hsv'); plt.axis('off'); plt.title("Hue Channel (Image 2)")
fig.add_subplot(rows, columns, 5); plt.imshow(s2, cmap='gray'); plt.axis('off'); plt.title("Saturation Channel (Image 2)")
fig.add_subplot(rows, columns, 6); plt.imshow(v2, cmap='gray'); plt.axis('off'); plt.title("Value Channel (Image 2)")

# show plot
plt.tight_layout()
plt.show()