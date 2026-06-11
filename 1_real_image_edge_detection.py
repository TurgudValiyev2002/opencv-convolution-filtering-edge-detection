from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_sample_images


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]


def conv2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    output = np.zeros_like(image, dtype=np.float32)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            output[row, col] = np.sum(padded[row : row + kh, col : col + kw] * kernel)
    return output


def normalize(image: np.ndarray) -> np.ndarray:
    lo = float(image.min())
    hi = float(image.max())
    return (image - lo) / (hi - lo + 1e-8)


def non_maximum_suppression(magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
    angle = (np.rad2deg(angle) + 180) % 180
    out = np.zeros_like(magnitude)
    rows, cols = magnitude.shape
    for row in range(1, rows - 1):
        for col in range(1, cols - 1):
            direction = angle[row, col]
            if (0 <= direction < 22.5) or (157.5 <= direction < 180):
                q, r = magnitude[row, col + 1], magnitude[row, col - 1]
            elif 22.5 <= direction < 67.5:
                q, r = magnitude[row + 1, col - 1], magnitude[row - 1, col + 1]
            elif 67.5 <= direction < 112.5:
                q, r = magnitude[row + 1, col], magnitude[row - 1, col]
            else:
                q, r = magnitude[row - 1, col - 1], magnitude[row + 1, col + 1]
            if magnitude[row, col] >= q and magnitude[row, col] >= r:
                out[row, col] = magnitude[row, col]
    return out


def connected_hysteresis(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    edges = strong.copy()
    changed = True
    while changed:
        changed = False
        grown = edges.copy()
        rows, cols = edges.shape
        for row in range(1, rows - 1):
            for col in range(1, cols - 1):
                if weak[row, col] and not edges[row, col] and edges[row - 1 : row + 2, col - 1 : col + 2].any():
                    grown[row, col] = True
                    changed = True
        edges = grown
    return edges.astype(np.float32)


def canny_like_edges(gray: np.ndarray) -> np.ndarray:
    gaussian = np.array(
        [
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1],
        ],
        dtype=np.float32,
    )
    gaussian = gaussian / gaussian.sum()
    smoothed = conv2d(gray, gaussian)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = sobel_x.T
    gx = conv2d(smoothed, sobel_x)
    gy = conv2d(smoothed, sobel_y)
    magnitude = normalize(np.sqrt(gx**2 + gy**2))
    suppressed = non_maximum_suppression(magnitude, np.arctan2(gy, gx))
    strong = suppressed >= np.quantile(suppressed, 0.94)
    weak = suppressed >= np.quantile(suppressed, 0.86)
    return connected_hysteresis(strong, weak)


def analyze_image(name: str, image: np.ndarray) -> dict[str, float | str]:
    gray = rgb_to_gray(image)
    blur = np.ones((3, 3), dtype=np.float32) / 9.0
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = sobel_x.T
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

    smoothed = conv2d(gray, blur)
    gx = conv2d(smoothed, sobel_x)
    gy = conv2d(smoothed, sobel_y)
    sobel = normalize(np.sqrt(gx**2 + gy**2))
    laplacian = normalize(np.abs(conv2d(smoothed, laplacian_kernel)))
    canny = canny_like_edges(gray)

    edge_threshold = 0.25
    row = {
        "image": name,
        "height": image.shape[0],
        "width": image.shape[1],
        "mean_gray_intensity": round(float(gray.mean()), 4),
        "sobel_edge_density": round(float((sobel > edge_threshold).mean()), 4),
        "laplacian_edge_density": round(float((laplacian > edge_threshold).mean()), 4),
        "canny_like_edge_density": round(float(canny.mean()), 4),
        "sobel_mean_strength": round(float(sobel.mean()), 4),
        "laplacian_mean_strength": round(float(laplacian.mean()), 4),
    }

    fig, axes = plt.subplots(1, 5, figsize=(14, 3.5))
    panels = [
        ("RGB image", image),
        ("Grayscale", gray),
        ("Sobel magnitude", sobel),
        ("Laplacian", laplacian),
        ("Canny-like edges", canny),
    ]
    for ax, (title, panel) in zip(axes, panels):
        ax.imshow(panel if panel.ndim == 3 else panel, cmap=None if panel.ndim == 3 else "gray")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Classical edge detection on real image: {name}", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{name}_edge_pipeline.png", dpi=180)
    plt.close(fig)
    return row


def plot_summary(summary: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 4))
    x = np.arange(len(summary))
    width = 0.25
    plt.bar(x - width, summary["sobel_edge_density"], width, label="Sobel")
    plt.bar(x, summary["laplacian_edge_density"], width, label="Laplacian")
    plt.bar(x + width, summary["canny_like_edge_density"], width, label="Canny-like")
    plt.xticks(x, summary["image"])
    plt.ylabel("Edge pixel share")
    plt.title("Edge Density by Real Image and Method")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "edge_density_comparison.png", dpi=180)
    plt.close()


def plot_readme_overview() -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    boxes = [
        ("Real photos", 0.14),
        ("Grayscale + blur", 0.38),
        ("Sobel / Laplacian\n/Canny-like filters", 0.64),
        ("Edge density\nand figures", 0.88),
    ]
    for text, xpos in boxes:
        ax.text(xpos, 0.55, text, ha="center", va="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.45", facecolor="#eef6ff", edgecolor="#336699"))
    for start, end in zip(boxes[:-1], boxes[1:]):
        ax.annotate("", xy=(end[1] - 0.12, 0.55), xytext=(start[1] + 0.12, 0.55), arrowprops=dict(arrowstyle="->", lw=2))
    ax.set_title("Real-image edge detection workflow", fontsize=15)
    fig.tight_layout()
    fig.savefig(ASSETS / "readme_project_overview.png", dpi=180)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    ASSETS.mkdir(exist_ok=True)
    dataset = load_sample_images()
    rows = []
    for name, image in zip(dataset.filenames, dataset.images):
        rows.append(analyze_image(Path(name).stem, image))
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "real_image_edge_summary.csv", index=False)
    plot_summary(summary)
    plot_readme_overview()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
