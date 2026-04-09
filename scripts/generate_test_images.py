"""
generate_test_images.py
Generates synthetic grayscale PGM (P5) test images for the CUDA Batch
Gaussian Blur project. Each image contains random noise plus a few simple
geometric shapes so that the blur effect is visually verifiable.

Usage:
    python3 scripts/generate_test_images.py \
        --output data/input \
        --count  100 \
        --width  512 \
        --height 512
"""

import argparse
import os
import struct
import random
import math


def make_image(width: int, height: int, seed: int) -> bytearray:
    """Return a width*height bytearray of grayscale pixel values."""
    rng = random.Random(seed)
    pixels = bytearray(rng.getrandbits(8) % 80 for _ in range(width * height))

    # Draw a few white circles
    for _ in range(5):
        cx = rng.randint(0, width - 1)
        cy = rng.randint(0, height - 1)
        r  = rng.randint(10, min(width, height) // 6)
        for y in range(max(0, cy - r), min(height, cy + r + 1)):
            for x in range(max(0, cx - r), min(width, cx + r + 1)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                    pixels[y * width + x] = rng.randint(180, 255)

    # Draw a few dark rectangles
    for _ in range(3):
        x0 = rng.randint(0, width - 20)
        y0 = rng.randint(0, height - 20)
        x1 = min(width,  x0 + rng.randint(15, 80))
        y1 = min(height, y0 + rng.randint(15, 80))
        val = rng.randint(0, 60)
        for y in range(y0, y1):
            for x in range(x0, x1):
                pixels[y * width + x] = val

    return pixels


def save_pgm(path: str, pixels: bytearray, width: int, height: int) -> None:
    with open(path, "wb") as f:
        header = f"P5\n{width} {height}\n255\n"
        f.write(header.encode("ascii"))
        f.write(pixels)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic PGM test images."
    )
    parser.add_argument("--output", required=True,
                        help="Output directory for .pgm files")
    parser.add_argument("--count",  type=int, default=20,
                        help="Number of images to generate (default: 20)")
    parser.add_argument("--width",  type=int, default=512,
                        help="Image width in pixels (default: 512)")
    parser.add_argument("--height", type=int, default=512,
                        help="Image height in pixels (default: 512)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Generating {args.count} images ({args.width}x{args.height}) "
          f"in '{args.output}'...")

    for i in range(args.count):
        pixels = make_image(args.width, args.height, seed=i)
        path   = os.path.join(args.output, f"img_{i:03d}.pgm")
        save_pgm(path, pixels, args.width, args.height)
        print(f"  Wrote {path}")

    print("Done.")


if __name__ == "__main__":
    main()
