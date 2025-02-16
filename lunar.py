import cv2
import numpy as np
import matplotlib.pyplot as plt

# 🔹 Step 1: Load Image from Correct Path
image_path = r"C:\Users\skbeh\OneDrive\Desktop\image test.png"  # Use raw string (r"") or double backslashes
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 🔹 Step 2: Check if Image Loaded Correctly
if img is None:
    raise ValueError("Error: Image not found. Check the path!")

print("Image shape:", img.shape)  # Should be (Height, Width)

# 🔹 Step 3: Display the Loaded Image
plt.figure(figsize=(6, 6))
plt.imshow(img, cmap='gray')
plt.title("Loaded Image")
plt.axis("off")
plt.show()

# 🔹 Step 4: Apply Contrast Enhancement
contrast_img = cv2.equalizeHist(img)

# 🔹 Step 5: Apply Edge Detection
edges = cv2.Canny(contrast_img, 50, 150)

# 🔹 Step 6: Display Results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.imshow(contrast_img, cmap='gray')
plt.title("Contrast Enhanced")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title("Edge Detection")
plt.axis("off")

print(plt.show())
