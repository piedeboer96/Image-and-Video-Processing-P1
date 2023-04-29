import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('images/birdie.jpg', cv2.IMREAD_GRAYSCALE)

# Convert image to float and normalize
img_float = img.astype(float) / 255.0

# Compute the FFT of the image
fft_img = np.fft.fft2(img_float)

# Shift the zero-frequency component to the center
shifted_fft_img = np.fft.fftshift(fft_img)

# Define the frequency component to be suppressed
noise_freq = 0.1  # Adjust this value based on the noise frequency

# Create the filter mask
mask = np.ones_like(shifted_fft_img)
mask[int(shifted_fft_img.shape[0]/2), int(noise_freq*shifted_fft_img.shape[1])] = 0

# Apply the filter mask
filtered_fft_img = shifted_fft_img * mask

# Shift the spectrum back
filtered_fft_img_shifted = np.fft.ifftshift(filtered_fft_img)

# Compute the inverse FFT to obtain the filtered image
filtered_img = np.abs(np.fft.ifft2(filtered_fft_img_shifted))

# Display the original and filtered images
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filtered_img, cmap='gray')
plt.title('Filtered Image')
plt.axis('off')

plt.tight_layout()
plt.show()