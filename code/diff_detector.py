"""
diff_detector.py
------------------
Core image-processing pipeline: given two near-identical images (A = reference,
B = test), detect and highlight regions that differ.

PIPELINE (classical CV, no training data required):
    1. Load + resize/align pair to identical dimensions.
    2. Convert to grayscale.
    3. Compute Structural Similarity (SSIM) map between A and B.
       - Why SSIM over raw pixel subtraction? SSIM compares local luminance,
         contrast, and structure in small windows, so it is far more robust to
         sensor noise, minor lighting shifts, and JPEG compression artifacts
         than naive |A-B| pixel differencing, which flags every noisy pixel.
    4. Threshold the (1 - SSIM) map with Otsu's method to get a binary "changed"
       mask (Otsu picks the threshold automatically from the image's own
       histogram, so we don't hand-tune a magic number per image).
    5. Morphological closing + opening to remove speckle noise and merge
       fragmented blobs that belong to the same defect.
    6. Contour detection on the cleaned mask -> bounding boxes.
    7. Filter tiny contours (area < MIN_AREA) as noise.
    8. Draw ROI bounding boxes on both images using Matplotlib and save results.

Run:
    python diff_detector.py            # processes all pairs in ../raw_samples
"""

import os
import glob
import json
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import matplotlib.patches as patches

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "raw_samples")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MIN_AREA = 40          # ignore contours smaller than this (pixels^2) - noise filter
SSIM_WIN = 7            # SSIM sliding-window size


def load_pair(path_a, path_b, target_size=None):
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)
    if img_a is None or img_b is None:
        raise FileNotFoundError(f"Could not read {path_a} or {path_b}")

    if target_size is None:
        # If sizes differ slightly (e.g. two separate camera captures), resize
        # B onto A's canvas so shapes line up for pixel/window-level comparison.
        target_size = (img_a.shape[1], img_a.shape[0])
    img_b = cv2.resize(img_b, target_size)
    img_a = cv2.resize(img_a, target_size)
    return img_a, img_b


def align_images(img_a, img_b, max_features=500, good_match_pct=0.15):
    """
    Optional ECC/feature-based alignment step for cases where the two shots
    were taken from a slightly different camera angle/position (common in a
    real inspection rig if the board isn't in a fixed jig). Uses ORB feature
    matching + homography. If alignment fails (e.g. too few matches, which is
    common on flat/low-texture industrial images), we fall back to the
    un-warped image rather than crashing.
    """
    try:
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(max_features)
        kp_a, des_a = orb.detectAndCompute(gray_a, None)
        kp_b, des_b = orb.detectAndCompute(gray_b, None)
        if des_a is None or des_b is None or len(kp_a) < 10 or len(kp_b) < 10:
            return img_b  # not enough features to safely align

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(matcher.match(des_a, des_b), key=lambda x: x.distance)
        n_good = max(int(len(matches) * good_match_pct), 4)
        matches = matches[:n_good]
        if len(matches) < 4:
            return img_b

        pts_a = np.float32([kp_a[m.queryIdx].pt for m in matches])
        pts_b = np.float32([kp_b[m.trainIdx].pt for m in matches])
        H_mat, _ = cv2.findHomography(pts_b, pts_a, cv2.RANSAC)
        if H_mat is None:
            return img_b
        aligned = cv2.warpPerspective(img_b, H_mat, (img_a.shape[1], img_a.shape[0]))
        return aligned
    except Exception:
        return img_b  # graceful fallback - never let alignment crash the pipeline


def detect_differences(img_a, img_b):
    """Returns: list of bounding boxes [(x, y, w, h), ...] and the diff heatmap."""
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    score, diff_map = ssim(gray_a, gray_b, win_size=SSIM_WIN, full=True)
    diff_map = (1 - diff_map)  # invert: high value = more different
    diff_norm = cv2.normalize(diff_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Otsu auto-thresholding on the normalized diff map
    _, mask = cv2.threshold(diff_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup: close small gaps, remove speckle noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, w, h))

    return boxes, diff_norm, mask, score


def draw_results(img_a, img_b, boxes, score, out_path, pair_name):
    """Render side-by-side result using Matplotlib with ROI bounding boxes,
    as required by the submission format."""
    img_a_rgb = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
    img_b_rgb = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, img, title in zip(axes, [img_a_rgb, img_b_rgb], ["Reference (A)", "Test Image (B)"]):
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")
        for (x, y, w, h) in boxes:
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="red", facecolor="none")
            ax.add_patch(rect)

    fig.suptitle(f"{pair_name}  |  SSIM similarity score: {score:.4f}  |  {len(boxes)} difference region(s) found",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_all_pairs():
    pair_files = sorted(glob.glob(os.path.join(SAMPLES_DIR, "pair_*_A.png")))
    summary = {}

    for path_a in pair_files:
        pair_name = os.path.basename(path_a).replace("_A.png", "")
        path_b = path_a.replace("_A.png", "_B.png")

        img_a, img_b = load_pair(path_a, path_b)
        img_b_aligned = align_images(img_a, img_b)

        boxes, diff_norm, mask, score = detect_differences(img_a, img_b_aligned)

        out_path = os.path.join(RESULTS_DIR, f"{pair_name}_result.png")
        draw_results(img_a, img_b_aligned, boxes, score, out_path, pair_name)

        summary[pair_name] = {
            "ssim_score": round(float(score), 4),
            "num_regions_detected": len(boxes),
            "boxes_xywh": boxes,
        }
        print(f"[{pair_name}] SSIM={score:.4f}  regions_found={len(boxes)}  -> {out_path}")

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {os.path.join(RESULTS_DIR, 'summary.json')}")


if __name__ == "__main__":
    process_all_pairs()
