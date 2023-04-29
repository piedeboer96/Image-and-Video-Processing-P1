import cv2 as cv
import numpy as np

# Load the image
img = cv.imread('images/sheep.jpg')
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4
ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))


# Convert the image to grayscale
gray = cv.cvtColor(res2, cv.COLOR_BGR2GRAY)

# Perform edge detection using Canny
edges = cv.Canny(gray, 100, 200)

# Create a black outline image
outline = np.zeros_like(res2)
outline[edges != 0] = (0, 0, 0)

# Combine the outline image with the segmented image
cartoon = cv.add(res2, outline)

# Display the cartoonified image
cv.imshow('Cartoonified Image', cartoon)
cv.waitKey(0)
cv.destroyAllWindows()