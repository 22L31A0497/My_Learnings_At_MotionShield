
# 🧠 OpenCV with Python

## 🧭 1. What is OpenCV?

### 🔍 Definition
- OpenCV (Open Source Computer Vision Library) is a cross-platform library for real-time computer vision.
- Available in Python, C++, and Java.

### 🧠 Purpose
- Helps extract insights from images and videos.
- Used in deep learning applications like object detection, face recognition, and image classification.

---

## 🛠️ 2. Course Setup

### 🐍 Python Version
- Recommended: Python 3.7 or above.
- Check version using `python --version`.

### 📦 Required Packages
- `opencv-contrib-python`: Full OpenCV package including community-contributed modules.
- `numpy`: For matrix and array operations.
- `seer`: A custom utility package built by the instructor to simplify workflows (used later in the course).

---

## 🖼️ 3. Reading Media Files

### 🖼️ Reading Images
- Images are read as pixel matrices.
- OpenCV treats images as arrays of pixel values (height × width × color channels).

### 🎥 Reading Videos
- Videos are read frame-by-frame using a loop.
- Can read from:
  - Webcam (using integer index like `0`)
  - Video file (using file path)

### 🧠 Key Concepts
- Each video frame is treated like an image.
- Reading media involves capturing and displaying frames using OpenCV methods.

---

## 📏 4. Rescaling and Resizing

### ⚙️ Why Resize?
- Large images/videos consume more memory and processing power.
- Resizing helps optimize performance and fit content to screen.

### 📐 Rescaling vs. Resolution Change
- **Rescaling**: Adjusts image size by a scale factor (e.g., 75% of original).
- **Resolution Change**: Sets explicit width and height (used for live video only).

### 🧠 Best Practices
- Always downscale for performance.
- Use scale factors for static media; use resolution settings for live feeds.

---

## 🖍️ 5. Drawing on Images

### 🧱 Creating Blank Images
- Blank images are arrays filled with zeros (black canvas).
- Useful for drawing shapes and testing visual features.

### 🟩 Drawing Shapes
- **Rectangle**: Defined by two corner points.
- **Circle**: Defined by center and radius.
- **Line**: Defined by start and end points.

### ✍️ Writing Text
- Text can be added using built-in fonts.
- Requires position, font type, scale, color, and thickness.

---

## 🧰 6. Basic Image Operations

### 🌑 Grayscale Conversion
- Converts color image (BGR) to grayscale.
- Useful for simplifying image data and reducing computation.

### 🌫️ Blurring
- Removes noise and smooths image.
- Gaussian blur is commonly used.

### 🧱 Edge Detection
- Canny edge detector identifies boundaries in images.
- Uses threshold values to detect strong and weak edges.

---

## 🧪 7. Image Morphology

### 🔍 Dilation
- Expands white regions in binary images.
- Makes edges thicker.

### 🧼 Erosion
- Shrinks white regions.
- Removes small noise and refines edges.

---

## ✂️ 8. Resizing and Cropping

### 📐 Resize
- Changes image dimensions to a fixed size.
- May distort aspect ratio if not handled carefully.

### ✂️ Crop
- Selects a region of interest (ROI) using array slicing.
- Useful for focusing on specific parts of an image.

---
