from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("results")

def conv2d(img, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    out = np.zeros_like(img, dtype=float)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    return out

def main():
    RESULTS.mkdir(exist_ok=True)
    img = np.zeros((96, 96), dtype=float)
    img[20:76, 24:72] = 0.7
    rr, cc = np.ogrid[:96, :96]
    img[(rr - 48) ** 2 + (cc - 48) ** 2 < 14 ** 2] = 1.0
    blur = np.ones((3, 3)) / 9
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = sobel_x.T
    smoothed = conv2d(img, blur)
    gx = conv2d(smoothed, sobel_x)
    gy = conv2d(smoothed, sobel_y)
    edges = np.sqrt(gx ** 2 + gy ** 2)
    pd.DataFrame([{"mean_intensity": img.mean(), "edge_mean": edges.mean(), "edge_max": edges.max()}]).to_csv(RESULTS / "image_edge_summary.csv", index=False)
    for name, arr in [("synthetic_image", img), ("smoothed_image", smoothed), ("edge_magnitude", edges)]:
        plt.figure(figsize=(4, 4))
        plt.imshow(arr, cmap="gray")
        plt.title(name.replace("_", " ").title())
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(RESULTS / f"{name}.png", dpi=160)
    plt.figure(figsize=(9, 3))
    for k, (title, arr) in enumerate([("Original", img), ("Smoothed", smoothed), ("Edges", edges)]):
        plt.subplot(1, 3, k + 1)
        plt.imshow(arr, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(RESULTS / "convolution_pipeline.png", dpi=160)
    print("saved convolution results")

if __name__ == "__main__":
    main()
