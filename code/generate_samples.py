"""
generate_samples.py
--------------------
Generates 5 pairs of near-identical images for testing the difference-detection
pipeline. Each pair = (image_A = "reference/original", image_B = "modified copy"
with a few deliberate differences introduced: added spots, lines, shape shifts,
color patches, or missing components).

Why synthetic generation instead of just downloading two random photos?
- We control ground truth (we KNOW exactly what changed and where), which lets
  us objectively verify precision/recall of the detector instead of eyeballing it.
- It mirrors the real industrial use-case this assignment is modeled on: SMT
  (Surface-Mount Technology) PCB inspection, where you compare a "golden
  reference" board image against a "device under test" image to catch defects
  (missing components, solder bridges, tombstoning, foreign particles, etc).
  So these synthetic boards double as a reasonable proxy for that domain.

Run:
    python generate_samples.py
Outputs:
    ../raw_samples/pair_1_A.png / pair_1_B.png ... pair_5_A.png / pair_5_B.png
"""

import os
import cv2
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "raw_samples")
os.makedirs(OUT_DIR, exist_ok=True)

W, H = 640, 480


def base_pcb_canvas():
    """A synthetic 'PCB-like' board: green background, grid of pads/traces."""
    img = np.full((H, W, 3), (40, 110, 40), dtype=np.uint8)  # PCB green
    # silkscreen-style grid
    for x in range(20, W, 40):
        cv2.line(img, (x, 0), (x, H), (30, 90, 30), 1)
    for y in range(20, H, 40):
        cv2.line(img, (0, y), (W, y), (30, 90, 30), 1)
    # component pads (consistent layout across all pairs)
    rng = np.random.RandomState(42)
    pads = []
    for i in range(14):
        cx = rng.randint(60, W - 60)
        cy = rng.randint(60, H - 60)
        r = rng.randint(10, 18)
        cv2.circle(img, (cx, cy), r, (200, 200, 200), -1)
        cv2.circle(img, (cx, cy), r, (120, 120, 120), 2)
        pads.append((cx, cy, r))
    # a few IC chips (black rectangles)
    chips = [(120, 100, 70, 40), (420, 320, 90, 50), (300, 150, 60, 60)]
    for (cx, cy, cw, ch) in chips:
        cv2.rectangle(img, (cx - cw // 2, cy - ch // 2), (cx + cw // 2, cy + ch // 2), (20, 20, 20), -1)
        cv2.rectangle(img, (cx - cw // 2, cy - ch // 2), (cx + cw // 2, cy + ch // 2), (200, 200, 0), 1)
    return img


def add_noise(img, sigma=2.5):
    """Realistic sensor noise so the two images are NEVER pixel-identical,
    forcing the pipeline to be robust to noise rather than relying on
    exact pixel equality."""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def make_pair(seed, diff_fn):
    rng_state = np.random.RandomState(seed)
    np.random.seed(seed)
    img_a = base_pcb_canvas()
    img_a = add_noise(img_a, sigma=2.0)

    img_b = img_a.copy()
    img_b, diff_desc = diff_fn(img_b)
    img_b = add_noise(img_b, sigma=2.0)  # independent noise realization
    return img_a, img_b, diff_desc


def diff_missing_component(img):
    """Simulate a missing solder pad (component not placed)."""
    cv2.circle(img, (300, 150 + 60), 14, (40, 110, 40), -1)  # paint over a pad with board color
    return img, "Missing component: one solder pad erased (simulates a part not placed on the board)"


def diff_foreign_particle(img):
    """Simulate dust/solder splash - small dark spot."""
    cv2.circle(img, (500, 200), 8, (10, 10, 10), -1)
    cv2.circle(img, (150, 350), 5, (10, 10, 10), -1)
    return img, "Foreign particles: two small dark spots added (simulates dust/solder splash contamination)"


def diff_solder_bridge(img):
    """Simulate a solder bridge - a thin bright line connecting two pads."""
    cv2.line(img, (280, 130), (320, 170), (210, 210, 210), 3)
    return img, "Solder bridge: a bright line drawn connecting two adjacent pads (simulates unwanted solder bridging)"


def diff_shifted_chip(img):
    """Simulate a tombstoned / shifted IC chip."""
    cx, cy, cw, ch = 420, 320, 90, 50
    cv2.rectangle(img, (cx - cw // 2, cy - ch // 2), (cx + cw // 2, cy + ch // 2), (40, 110, 40), -1)  # erase original
    cv2.rectangle(img, (cx - cw // 2 + 15, cy - ch // 2 + 8), (cx + cw // 2 + 15, cy + ch // 2 + 8), (20, 20, 20), -1)
    cv2.rectangle(img, (cx - cw // 2 + 15, cy - ch // 2 + 8), (cx + cw // 2 + 15, cy + ch // 2 + 8), (200, 200, 0), 1)
    return img, "Misaligned component: IC chip shifted ~15-20px from its correct position (simulates placement drift)"


def diff_discoloration(img):
    """Simulate a burn mark / discoloration patch."""
    overlay = img.copy()
    cv2.rectangle(overlay, (60, 380), (160, 440), (0, 60, 140), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    return img, "Discoloration: a brownish patch blended onto the board (simulates a burn/heat-damage mark)"


PAIR_DEFS = [
    (1, diff_missing_component),
    (2, diff_foreign_particle),
    (3, diff_solder_bridge),
    (4, diff_shifted_chip),
    (5, diff_discoloration),
]

if __name__ == "__main__":
    manifest = []
    for idx, fn in PAIR_DEFS:
        a, b, desc = make_pair(seed=idx * 7, diff_fn=fn)
        cv2.imwrite(os.path.join(OUT_DIR, f"pair_{idx}_A.png"), a)
        cv2.imwrite(os.path.join(OUT_DIR, f"pair_{idx}_B.png"), b)
        manifest.append(f"pair_{idx}: {desc}")
        print(f"[OK] pair_{idx} -> {desc}")

    with open(os.path.join(OUT_DIR, "manifest.txt"), "w") as f:
        f.write("\n".join(manifest))
    print(f"\nAll 5 pairs written to {os.path.abspath(OUT_DIR)}")
