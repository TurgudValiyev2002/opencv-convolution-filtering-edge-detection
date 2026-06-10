# Report: Convolution Filtering and Edge Detection

## Motivation

We implemented convolution to understand how classical computer vision filters process images.

## Dataset / Problem

The experiment used a controlled grayscale image with simple geometric shapes.

## Method

We applied mean filtering for smoothing and Sobel filters for edge detection.

## Hyperparameters

The image size was 96x96. The blur kernel was 3x3, and the edge filters were Sobel X and Sobel Y.

## Results

The mean edge magnitude was 0.1533, and the maximum edge magnitude was 2.3594. Figures show the original image, smoothed image, and detected edges.

## Interpretation

Edges appear where pixel intensity changes sharply. This confirms that Sobel filters capture object boundaries.

## Conclusion

Convolution is a simple but powerful operation. Understanding it makes modern CNNs easier to understand.
