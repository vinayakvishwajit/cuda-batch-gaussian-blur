# CUDA Batch Gaussian Blur

A GPU-accelerated image processing pipeline that applies Gaussian blur to a batch of grayscale images using the **CUDA NPP (NVIDIA Performance Primitives)** library.

Processes hundreds of images in a single execution — each image is uploaded to the GPU, filtered using `nppiFilterGauss_8u_C1R`, and written back to disk. Per-image timing and a total wall-clock summary are logged to stdout.

---

## Project Structure

```
.
├── src/
│   └── image_processor.cu   # Main CUDA source
├── scripts/
│   └── generate_test_images.py  # Generates synthetic .pgm test images
├── data/
│   ├── input/               # Place input .pgm images here
│   └── output/              # Blurred output images written here
├── results/                 # Screenshots / logs proving execution
├── bin/                     # Compiled binary (created by make)
├── Makefile
├── run.sh
├── INSTALL
└── README.md
```

---

## Requirements

- Linux (tested on Ubuntu 20.04+)
- CUDA Toolkit 11.x or 12.x
- NVIDIA GPU (compute capability 6.0 or higher)
- Python 3 + NumPy (only for generating test images)

---

## Build

```bash
make
```

This compiles `src/image_processor.cu` and produces `bin/image_processor`.

To clean:

```bash
make clean
```

---

## Run

### Option 1 — Quick start with `run.sh`

```bash
chmod +x run.sh
./run.sh
```

This builds the project, generates 20 synthetic 512×512 test images if none exist, then runs the processor.

You can also pass custom arguments:

```bash
./run.sh data/input data/output 7
#         ^input     ^output    ^mask size (3/5/7/9/11)
```

### Option 2 — Run the binary directly

```bash
./bin/image_processor --input data/input --output data/output --mask 5
```

**CLI arguments:**

| Argument   | Description                          | Default |
|------------|--------------------------------------|---------|
| `--input`  | Directory containing `.pgm` images   | required |
| `--output` | Directory to write blurred images    | required |
| `--mask`   | Gaussian mask size (3/5/7/9/11)      | `5`     |
| `--help`   | Print usage                          | —       |

---

## Generating Test Images

If you don't have `.pgm` images, generate synthetic ones:

```bash
python3 scripts/generate_test_images.py --output data/input --count 100 --width 256 --height 256
```

Or for a smaller set of large images:

```bash
python3 scripts/generate_test_images.py --output data/input --count 10 --width 2048 --height 2048
```

---

## Example Output

```
[INFO] GPU: NVIDIA GeForce RTX 3060 (CUDA 8.6)
[INFO] Input dir  : data/input
[INFO] Output dir : data/output
[INFO] Mask size  : 5x5
[INFO] -------------------------------------------
[INFO] Found 100 image(s) to process.
[  1/100] img_000.pgm                     512x 512  1.83 ms
[  2/100] img_001.pgm                     512x 512  0.74 ms
[  3/100] img_002.pgm                     512x 512  0.71 ms
...
[100/100] img_099.pgm                     512x 512  0.69 ms
[INFO] -------------------------------------------
[INFO] Done. 100 succeeded, 0 failed. Total wall time: 312.47 ms
```

---

## Algorithm

The program uses **`nppiFilterGauss_8u_C1R`** from the NVIDIA NPP library.

NPP's Gaussian filter applies a 2D Gaussian convolution kernel of size N×N (where N ∈ {3, 5, 7, 9, 11}) to each pixel using the GPU. The kernel weights are precomputed by NPP based on the mask size. This is significantly faster than a CPU-side convolution for large batches because:

1. All pixel operations are independent — the GPU can process thousands in parallel.
2. NPP internally uses optimized shared memory tiling to minimize global memory traffic.
3. The entire image stays on the GPU between upload and download, eliminating redundant transfers.

---

## Lessons Learned

- NPP's pitched memory (from `nppiMalloc_8u_C1`) requires row-by-row `cudaMemcpy` for correct alignment — using a flat memcpy caused silent data corruption until row stride was accounted for.
- For batches of small images, the per-image H2D/D2H transfer overhead dominates GPU compute time. Fusing multiple images into a single batch transfer would improve throughput significantly.
- The `nppiFilterGauss_8u_C1R` ROI must exactly match the allocated image dimensions, otherwise NPP returns `NPP_SIZE_ERROR` without a descriptive message.

---

## License

GPL-3.0. See [LICENSE](LICENSE).
