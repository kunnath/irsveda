import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys

# Load the image
image_path = '4.png'
image = cv2.imread(image_path)

if image is None:
    print(f"Could not read image {image_path}")
    sys.exit(1)

# Convert BGR to RGB for display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display basic image info
print(f"Image shape: {image.shape}")
print(f"Image type: {image.dtype}")

# Show the image
plt.figure(figsize=(10, 8))
plt.imshow(image_rgb)
plt.title('4.png')
plt.axis('off')
plt.savefig('4_analysis.png')

# Display color channels separately
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image[:, :, 0], cmap='Blues')
plt.title('Blue Channel')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(image[:, :, 1], cmap='Greens')
plt.title('Green Channel')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(image[:, :, 2], cmap='Reds')
plt.title('Red Channel')
plt.axis('off')
plt.savefig('4_channels.png')

print("Analysis complete. Check 4_analysis.png and 4_channels.png")
