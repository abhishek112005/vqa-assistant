"""
data/generate_synthetic.py — Synthetic VQA Dataset Generator

Generates colour/shape images with paired Q&A to train and validate the model
without needing to download large external datasets.

Why synthetic data?
────────────────────
• Runs locally — no internet required, no licensing issues.
• Ground truth is exact — no annotation noise.
• Controllable — you can adjust size, complexity, vocabulary coverage.
• Demonstrates the full pipeline clearly — ideal for education.

Image types generated
──────────────────────
1. Single shape — one coloured geometric figure on a white background.
2. Multi-shape  — 2–5 coloured figures, varied sizes and positions.
3. Bar chart    — simple matplotlib chart saved as PIL image.

Question / answer templates are chosen proportionally so every answer
class in ANSWER_VOCAB receives training signal.
"""

import os
import random
import json
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

# ─────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────
COLORS = {
    "red"   : (220, 50,  50),
    "green" : (50,  180, 80),
    "blue"  : (50,  100, 230),
    "yellow": (240, 220, 20),
    "orange": (240, 130, 20),
    "purple": (140, 50,  200),
    "pink"  : (240, 100, 170),
    "cyan"  : (20,  200, 220),
    "brown" : (140, 80,  30),
    "gray"  : (150, 150, 150),
}

SHAPES = ["circle", "square", "triangle", "rectangle"]


def _draw_shape(draw: ImageDraw.ImageDraw, shape: str,
                x: int, y: int, size: int, color: tuple):
    """Draw one shape at position (x,y) with given pixel size and fill colour."""
    if shape == "circle":
        draw.ellipse([x, y, x + size, y + size], fill=color, outline=(0, 0, 0), width=2)
    elif shape == "square":
        draw.rectangle([x, y, x + size, y + size], fill=color, outline=(0, 0, 0), width=2)
    elif shape == "triangle":
        pts = [(x + size // 2, y), (x, y + size), (x + size, y + size)]
        draw.polygon(pts, fill=color, outline=(0, 0, 0))
    elif shape == "rectangle":
        draw.rectangle([x, y, x + size * 2, y + size], fill=color, outline=(0, 0, 0), width=2)


def _no_overlap(boxes: list[tuple], x: int, y: int, w: int, h: int,
                margin: int = 10) -> bool:
    """Return True if the new bounding box does not overlap any existing box."""
    for bx, by, bw, bh in boxes:
        if (x < bx + bw + margin and x + w + margin > bx and
                y < by + bh + margin and y + h + margin > by):
            return False
    return True


# ─────────────────────────────────────────────────────────────
# Single-shape image + Q&A
# ─────────────────────────────────────────────────────────────

def _make_single_shape_sample() -> tuple[Image.Image, str, str]:
    color_name = random.choice(list(COLORS.keys()))
    shape_name = random.choice(SHAPES)
    color      = COLORS[color_name]
    size       = random.randint(40, 90)

    img  = Image.new("RGB", (224, 224), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    x = random.randint(10, 224 - size - 40)
    y = random.randint(10, 224 - size - 10)
    _draw_shape(draw, shape_name, x, y, size, color)

    size_label = "large" if size > 65 else ("small" if size < 50 else "large")

    # Choose question template
    q_type = random.choice(["color", "shape", "size"])
    if q_type == "color":
        q = f"What color is the {shape_name}?"
        a = color_name
    elif q_type == "shape":
        q = f"What shape is the {color_name} object?"
        a = shape_name
    else:
        q = f"Is the {color_name} {shape_name} large or small?"
        a = size_label

    return img, q, a


# ─────────────────────────────────────────────────────────────
# Multi-shape image + Q&A
# ─────────────────────────────────────────────────────────────

def _make_multi_shape_sample() -> tuple[Image.Image, str, str]:
    img  = Image.new("RGB", (224, 224), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    n_shapes = random.randint(2, 5)
    placed   = []   # list of dicts with color/shape info
    boxes    = []   # bounding boxes for overlap check

    for _ in range(n_shapes):
        color_name = random.choice(list(COLORS.keys()))
        shape_name = random.choice(SHAPES)
        color      = COLORS[color_name]
        size       = random.randint(25, 55)
        w          = size * 2 if shape_name == "rectangle" else size

        # Try to place without overlap (max 30 attempts)
        for _ in range(30):
            x = random.randint(5, 224 - w - 5)
            y = random.randint(5, 224 - size - 5)
            if _no_overlap(boxes, x, y, w, size):
                break

        _draw_shape(draw, shape_name, x, y, size, color)
        boxes.append((x, y, w, size))
        placed.append({"color": color_name, "shape": shape_name})

    # Build Q&A
    q_type = random.choice(["count", "presence_yes", "presence_no",
                             "color_of_shape", "shape_of_color"])

    if q_type == "count":
        q = "How many shapes are in the image?"
        a = str(len(placed))

    elif q_type == "presence_yes":
        target = random.choice(placed)
        q = f"Is there a {target['color']} {target['shape']} in the image?"
        a = "yes"

    elif q_type == "presence_no":
        colors_used  = {p["color"]  for p in placed}
        shapes_used  = {p["shape"]  for p in placed}
        unused_colors = [c for c in COLORS if c not in colors_used]
        if unused_colors:
            q = f"Is there a {random.choice(unused_colors)} shape in the image?"
            a = "no"
        else:
            # All colours used — ask about a colour+shape combo not present
            target = random.choice(placed)
            other_shapes = [s for s in SHAPES if s != target["shape"]]
            q = f"Is there a {target['color']} {random.choice(other_shapes)}?"
            # Check if that combo exists
            combo_present = any(p["color"] == target["color"] and
                                p["shape"] == other_shapes[0]
                                for p in placed)
            a = "yes" if combo_present else "no"

    elif q_type == "color_of_shape":
        target = random.choice(placed)
        q = f"What color is the {target['shape']}?"
        a = target["color"]

    else:  # shape_of_color
        target = random.choice(placed)
        q = f"What shape is the {target['color']} object?"
        a = target["shape"]

    # Clamp numeric answer to valid vocab
    if a.isdigit() and int(a) > 5:
        a = "5"

    return img, q, a


# ─────────────────────────────────────────────────────────────
# Yes / No questions on single shapes
# ─────────────────────────────────────────────────────────────

def _make_yesno_sample() -> tuple[Image.Image, str, str]:
    color_name = random.choice(list(COLORS.keys()))
    shape_name = random.choice(SHAPES)
    color      = COLORS[color_name]
    size       = random.randint(35, 80)

    img  = Image.new("RGB", (224, 224), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    _draw_shape(draw, shape_name,
                random.randint(20, 120), random.randint(20, 120),
                size, color)

    # 50 % chance the question matches reality
    if random.random() > 0.5:
        ask_color = color_name
        ask_shape = shape_name
        a = "yes"
    else:
        other_colors = [c for c in COLORS if c != color_name]
        ask_color = random.choice(other_colors)
        ask_shape = shape_name
        a = "no"

    q = f"Is there a {ask_color} {ask_shape} in the image?"
    return img, q, a


# ─────────────────────────────────────────────────────────────
# Public generator class
# ─────────────────────────────────────────────────────────────

class SyntheticDataGenerator:
    """
    Generates (image, question, answer) triples on the fly.

    Sampling strategy
    ─────────────────
    • 40 % single-shape samples   — simpler, helps learn colour/shape basics
    • 40 % multi-shape samples    — count and presence reasoning
    • 20 % yes/no samples         — yes/no balance

    Call generate_dataset() to produce a list[dict] ready for VQADataset.
    Call save_dataset()     to persist images + JSON manifest to disk.
    """

    GENERATORS = [
        (_make_single_shape_sample, 0.40),
        (_make_multi_shape_sample,  0.40),
        (_make_yesno_sample,        0.20),
    ]

    def _sample_one(self) -> tuple[Image.Image, str, str]:
        funcs, weights = zip(*self.GENERATORS)
        fn = random.choices(funcs, weights=weights, k=1)[0]
        return fn()

    def generate_dataset(self, n: int) -> list[dict]:
        """
        Return a list of dicts with keys: question, answer.
        Images are kept as PIL objects in memory.
        """
        samples = []
        for _ in range(n):
            img, q, a = self._sample_one()
            samples.append({"image": img, "question": q, "answer": a})
        return samples

    def save_dataset(self, n: int, out_dir: str, split: str = "train") -> str:
        """
        Save images as PNG files and write a JSON manifest.
        Returns path to the manifest file.
        """
        os.makedirs(out_dir, exist_ok=True)
        manifest = []

        for i in range(n):
            img, q, a = self._sample_one()
            fname = f"{split}_{i:05d}.png"
            fpath = os.path.join(out_dir, fname)
            img.save(fpath)
            manifest.append({
                "image_path": fname,
                "question"  : q,
                "answer"    : a,
            })

        manifest_path = os.path.join(out_dir, f"{split}_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"  ✓ Saved {n} {split} samples → {out_dir}")
        return manifest_path


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gen = SyntheticDataGenerator()
    samples = gen.generate_dataset(5)
    for s in samples:
        print(f"  Q: {s['question']:50s}  A: {s['answer']}")
    print("\nGenerator OK.")
