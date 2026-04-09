# Makefile for CUDA Batch Gaussian Blur
# Requires: CUDA Toolkit (nvcc), CUDA NPP library

NVCC        := nvcc
BIN_DIR     := bin
SRC_DIR     := src
TARGET      := $(BIN_DIR)/image_processor

# CUDA architecture flags — covers Maxwell through Hopper
# Adjust sm_XX to match your GPU if needed
ARCH_FLAGS  := -gencode arch=compute_60,code=sm_60 \
               -gencode arch=compute_70,code=sm_70 \
               -gencode arch=compute_75,code=sm_75 \
               -gencode arch=compute_80,code=sm_80 \
               -gencode arch=compute_86,code=sm_86 \
               -gencode arch=compute_89,code=sm_89

NVCC_FLAGS  := -O2 -std=c++14 $(ARCH_FLAGS)

# NPP libraries needed for filtering functions
NPP_LIBS    := -lnppc -lnppif -lnppicc -lnppig

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
