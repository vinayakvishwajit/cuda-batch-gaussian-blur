# Makefile for CUDA Batch Gaussian Blur
# Requires: CUDA Toolkit (nvcc), CUDA NPP library

NVCC        := nvcc
BIN_DIR     := bin
SRC_DIR     := src
TARGET      := $(BIN_DIR)/image_processor

# CUDA architecture flags — compatible with CUDA 10.x and 11.x lab environments
# compute_35 kept for older Coursera lab GPUs (deprecated but functional)
ARCH_FLAGS  := -gencode arch=compute_35,code=compute_35 \
               -gencode arch=compute_60,code=sm_60 \
               -gencode arch=compute_70,code=sm_70 \
               -gencode arch=compute_75,code=sm_75

NVCC_FLAGS  := -O2 -std=c++14 $(ARCH_FLAGS)

# NPP static libraries — matches Coursera lab environment
NPP_LIBS    := -lnppisu_static -lnppif_static -lnppc_static -lculibos

LIBS        := $(NPP_LIBS) -lcudart

SRCS        := $(SRC_DIR)/image_processor.cu

.PHONY: all clean

all: $(BIN_DIR) $(TARGET)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(LIBS)
	@echo "Build successful: $@"

clean:
	rm -f $(TARGET)
	@echo "Cleaned."
