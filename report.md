# One-Page Report: Real-Image Edge Detection

## Motivation

We wanted to study edge detection on real images instead of simple artificial shapes. This makes the filtering behavior more realistic because real photos contain texture, lighting variation, and clutter.

## Dataset

The experiment uses two real scikit-learn sample photos: `china` and `flower`. Each image is an RGB photo with size 427x640.

## Method

We implemented 2D convolution from scratch. Each image was converted to grayscale, smoothed, and processed with Sobel, Laplacian, and Canny-like edge detection.

## Results

The China image had Sobel edge density 0.1004 and Canny-like edge density 0.0871. The flower image had Sobel edge density 0.0403 and Canny-like edge density 0.0688.

## Interpretation

The China image contains more strong boundaries and scene structure, so Sobel detects more edge pixels. The Canny-like method is more selective because it suppresses weak non-maximum responses and keeps connected edge chains.

## Conclusion

Classical filters remain useful for understanding visual structure. Real images show why thresholding, smoothing, and edge connectivity matter.
