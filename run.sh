#!/usr/bin/env bash
# run.sh — Build and run the CUDA Batch Gaussian Blur processor
#
# Usage:
#   ./run.sh                          # uses default data/input and data/output
#   ./run.sh <input_dir> <output_dir> [mask_size]
#
# Examples:
#   ./run.sh data/input data/output 7
#   ./run.sh /tmp/images /tmp/blurred 5

INPUT_DIR="${1:-data/input}"
OUTPUT_DIR="${2:-data/output}"
MASK_SIZE="${3:-5}"

# ---- Build ----
echo "[run.sh] Building..."
make -j"$(nproc 2>/dev/null || echo 4)" || { echo "[run.sh] Build failed."; exit 1; }

# ---- Prepare directories ----
mkdir -p "${INPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# ---- Check for input images and generate if needed ----
PGM_COUNT=0
for f in "${INPUT_DIR}"/*.pgm; do
  [ -f "$f" ] && PGM_COUNT=$((PGM_COUNT + 1))
done

if [ "${PGM_COUNT}" -eq 0 ]; then
  echo "[run.sh] No .pgm files found in ${INPUT_DIR}."
  echo "[run.sh] Generating synthetic test images..."
  python3 scripts/generate_test_images.py \
      --output "${INPUT_DIR}" \
      --count 20 \
      --width 512 \
      --height 512 || { echo "[run.sh] Image generation failed."; exit 1; }
fi

# ---- Run ----
echo "[run.sh] Running image_processor..."
./bin/image_processor \
    --input  "${INPUT_DIR}" \
    --output "${OUTPUT_DIR}" \
    --mask   "${MASK_SIZE}" || { echo "[run.sh] image_processor failed."; exit 1; }

echo "[run.sh] Done. Output written to: ${OUTPUT_DIR}"
