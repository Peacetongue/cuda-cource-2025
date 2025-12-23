# Makefile для cuBLAS & cuSPARSE примеров

# Компилятор и флаги
NVCC = nvcc
CFLAGS = -O2 -std=c++14
ARCH = -arch=sm_70
INCLUDE = -I./include
LIBS = -lcublas -lcusparse -lcusolver -lcudnn -lnvinfer -lnvonnxparser

# Директории
SRC_DIR = src
BUILD_DIR = build

# Исходные файлы
SOURCES = $(SRC_DIR)/main.cu $(SRC_DIR)/gemm_example.cu $(SRC_DIR)/spmv_example.cu $(SRC_DIR)/solver_example.cu $(SRC_DIR)/thrust_example.cu $(SRC_DIR)/cudnn_mlp_example.cu $(SRC_DIR)/tensorrt_example.cu
TARGET = cublas_cusparse_demo

# Цели
.PHONY: all clean run run-gemm run-spmv run-solver run-thrust run-cudnn run-tensorrt

all: $(TARGET)

$(TARGET): $(SOURCES)
	@echo "Compiling $(TARGET)..."
	$(NVCC) $(CFLAGS) $(ARCH) $(INCLUDE) -o $(TARGET) $(SOURCES) $(LIBS)
	@echo "Build complete!"

clean:
	@echo "Cleaning..."
	rm -f $(TARGET)
	rm -rf $(BUILD_DIR)
	@echo "Clean complete!"

run: $(TARGET)
	@echo "Running all examples..."
	./$(TARGET) all

run-gemm: $(TARGET)
	@echo "Running cuBLAS GEMM example..."
	./$(TARGET) gemm

run-spmv: $(TARGET)
	@echo "Running cuSPARSE SpMV example..."
	./$(TARGET) spmv

run-solver: $(TARGET)
	@echo "Running cuSOLVER sparse solver example..."
	./$(TARGET) solver

run-thrust: $(TARGET)
	@echo "Running Thrust parallel algorithms example..."
	./$(TARGET) thrust

run-cudnn: $(TARGET)
	@echo "Running cuDNN MLP example..."
	./$(TARGET) cudnn

run-tensorrt: $(TARGET)
	@echo "Running TensorRT inference example..."
	./$(TARGET) tensorrt

help:
	@echo "Available targets:"
	@echo "  make              - Build the project"
	@echo "  make clean        - Remove built files"
	@echo "  make run          - Run all examples"
	@echo "  make run-gemm     - Run only GEMM example"
	@echo "  make run-spmv     - Run only SpMV example"
	@echo "  make run-solver   - Run only sparse solver example"
	@echo "  make run-thrust   - Run only Thrust example"
	@echo "  make run-cudnn    - Run only cuDNN MLP example"
	@echo "  make run-tensorrt - Run only TensorRT inference example"
	@echo "  make help         - Show this help message"
