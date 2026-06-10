# Convolution Filtering and Edge Detection

## Motivation

Convolution is one of the core operations in computer vision. Before using large vision models, it is important to understand how simple filters detect structure in an image.

## Project Goal

We implemented convolution from scratch and used it for smoothing and edge detection.

## Dataset / Problem

The input is a controlled grayscale image containing simple shapes. This makes the effect of each filter easy to see.

## Tools

Python, NumPy, pandas, and matplotlib.

## Method

We applied a 3x3 mean filter for smoothing. Then we applied Sobel X and Sobel Y filters and combined them into an edge-magnitude image.

## Hyperparameters

- Image size: 96x96
- Blur kernel: 3x3 mean filter
- Edge filters: Sobel X and Sobel Y

## Results

The result summary reported:

| Metric | Value |
|---|---:|
| Mean image intensity | 0.2345 |
| Mean edge magnitude | 0.1533 |
| Maximum edge magnitude | 2.3594 |

The result folder contains the original image, smoothed image, edge image, combined pipeline figure, and a CSV summary.

## Interpretation

The edge magnitude is highest where the image intensity changes sharply, especially around the rectangle and circle boundaries. Smoothing reduces small local changes before edge detection.

## Conclusion

This project shows the basic idea behind many computer vision pipelines: filters transform raw pixels into more useful visual signals.

## How To Run

```bash
pip install -r requirements.txt
python 1_convolution_edge_detection.py
```
