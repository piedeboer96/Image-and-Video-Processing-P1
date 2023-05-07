import cv2
import numpy as np

def add_gaussian_noise(img, mean, std_dev):
    # noise matrix
    noise = np.zeros_like(img, dtype=np.float64)

    # normalize the grayscale image to a range of [0, 1]
    result = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    # generate gaussian noise
    cv2.randn(noise, mean, std_dev)

    # add noise to the result
    result = result + noise

    # normalize the result to a range of [0, 1]
    result = cv2.normalize(result, None, 0.0, 1.0, cv2.NORM_MINMAX)

    return result

def remove_gaussian_noise(img, kernel_size,sigma):
    # convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply median filter to remove noise
    median = cv2.medianBlur(gray, kernel_size,sigma)

    return median

# read the image

img = cv2.imread('images/birdie.jpg', cv2.COLOR_BGR2GRAY) 

# add Gaussian noise
noisy_img = add_gaussian_noise(img, 0, 0.1)

# remove Gaussian noise using a median filter
denoised_img = remove_gaussian_noise(noisy_img, 5,0)

# display the original, noisy, and denoised images side by side
# combined = np.concatenate((img, noisy_img, denoised_img), axis=1)
# cv2.imshow('Original / Noisy / Denoised', combined)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
