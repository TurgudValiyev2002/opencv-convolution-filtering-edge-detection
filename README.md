# Convolution Filtering and Edge Detection

## 1. Motivation

This computer vision lab implements convolution and Sobel edge detection from scratch with NumPy. OpenCV is named in the project idea, but the local environment has no cv2 package, so the implementation is transparent and dependency-light.

## 2. Project Goal

Build a small, reproducible AI research lab with clear outputs and honest limitations.

## 3. Dataset, Paper, Or Problem Description

Dataset/problem: a generated grayscale image with rectangle and circle shapes.

## 4. Tools

Tools: Python, NumPy, pandas, matplotlib.

## 5. Models Or Methods

Method: 3x3 mean blur, Sobel X/Y filters, edge magnitude.

## 6. Hyperparameters When Relevant

Hyperparameters: 96x96 image, 3x3 blur kernel, Sobel kernels.

## 7. Results

Results include original, smoothed, edge, pipeline figures, and summary CSV.

## 8. Interpretation Of Results

Interpretation: large edge magnitude appears where intensity changes sharply.

## 9. Conclusion

Conclusion: convolution is the core operation behind many classical and deep CV systems.

## 10. How To Run

```bash
pip install -r requirements.txt
python 1_*.py
```
