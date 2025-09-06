# Car Damage Instance Segmentation - Stage 1 Learning Guide

Welcome! 👋 This guide explains the **Stage 1 foundational topics** for Car Damage Instance Segmentation using **YOLOv8m**. It’s written in **easy-to-understand words** so even beginners can follow along. Each section has:

* **Simple explanation**
* **Python code example**
* **Best YouTube resource link**

Use this as your **GitHub README.md** or export it as a PDF for reference.

---

## 1. Digital Image Basics

### 📘 Explanation

* **Pixels:** Images are made of very tiny squares called pixels. Each pixel stores a color value. In **grayscale**, it’s a number between 0 (black) and 255 (white). In **RGB**, every pixel has three numbers: \[Red, Green, Blue].
* **Resolution:** Number of pixels in an image (width × height). Higher resolution → more details.
* **Types of Images:**

  * Black & White (binary): Only 0 or 1
  * Grayscale: 0–255 shades
  * RGB: Full color
  * Binary Mask: Used in segmentation → 0 = background, 1 = object

### 💻 Sample Code

```python
import cv2
import numpy as np

# Load an image
img = cv2.imread('car.jpg')
print("Image shape (height, width, channels):", img.shape)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("Grayscale pixel value at (50, 50):", gray[50, 50])

# Display grayscale image
cv2.imshow('Grayscale Image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 🎥 YouTube Resource

* [How Computers See Images (RGB & Pixels)](https://www.youtube.com/watch?v=ixyXUgmFfE0)

---

## 2. Segmentation Concepts

### 📘 Explanation

* **Object Detection:** Finds objects with rectangles (bounding boxes).
* **Semantic Segmentation:** Colors every pixel but does not separate objects of the same type.
* **Instance Segmentation:** Draws exact **pixel masks for each object separately** (e.g., 2 dents = 2 masks).
* **Why YOLOv8m?** It can detect and segment damages like **scratches, dents, cracks** precisely.
* **Masks:** Binary images showing where the damage is (1 = damage, 0 = background).

### 💻 Sample Code

```python
import numpy as np
import cv2

# Create a blank mask
mask = np.zeros((512, 512), dtype=np.uint8)

# Draw a white rectangle (simulate damage area)
mask[100:200, 150:250] = 1

# Show mask (multiply by 255 for display)
cv2.imshow('Damage Mask', mask * 255)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 🎥 YouTube Resource

* [Instance Segmentation Explained Simply](https://www.youtube.com/watch?v=nDPWyw9kGUs)

---

## 3. Training Fundamentals

### 📘 Explanation

* **Epoch:** One complete pass of the dataset during training.
* **Epoch Count:** Normally 50–100 gives good results.
* **Loss Functions:**

  * **Box Loss:** How well the bounding box fits.
  * **Class Loss:** How well the class (scratch, dent, etc.) is predicted.
  * **Segmentation Loss:** How well the mask matches the damage shape.

### 💻 Sample Code

```python
from ultralytics import YOLO

# Load YOLOv8m pretrained segmentation model
model = YOLO("yolov8m-seg.pt")

# Train model on custom dataset (dataset.yaml)
model.train(data="dataset.yaml", epochs=60, imgsz=640)
```

### 🎥 YouTube Resource

* [YOLOv8 Training Step-by-Step](https://www.youtube.com/watch?v=Fh6czYZp6Dg)

---

## 4. Damage Percentage Calculation

### 📘 Explanation

We can calculate **how much of the car is damaged** by comparing mask pixels:

$Damage% = (Mask Pixels ÷ Car Pixels) × 100$

* **Mask Pixels:** Pixels marked as damage.
* **Car Pixels:** Pixels belonging to the car body.

### 💻 Sample Code

```python
import cv2
import numpy as np

# Load masks (grayscale)
damage_mask = cv2.imread('damage_mask.png', 0)
car_mask = cv2.imread('car_mask.png', 0)

# Count pixels
damage_area = np.count_nonzero(damage_mask)
car_area = np.count_nonzero(car_mask)

# Calculate damage %
damage_percent = (damage_area / car_area) * 100
print(f'Damage Percentage: {damage_percent:.2f}%')
```

### 🎥 YouTube Resource

* [Segmentation and Damage Area Calculation](https://www.youtube.com/watch?v=MQ-uR5F5Fbs)

---

## ✅ How to Use This Guide

1. Read each section carefully.
2. Run the code examples with your own images.
3. Watch the YouTube videos for better visualization.
4. Use this as a **reference guide for Stage 1 project work**.

---

✨ This completes **Stage 1 Learning Guide** for Car Damage Instance Segmentation. Perfect to upload as your `README.md` on GitHub!
