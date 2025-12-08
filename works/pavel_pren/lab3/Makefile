CC = nvcc
CFLAGS = -O2
TARGET = sobel

.PHONY: all build run clean

all: build

build: $(TARGET)

$(TARGET): sobel.cu
	$(CC) $(CFLAGS) -o $(TARGET) sobel.cu

run: build
	./$(TARGET)

clean:
	rm -f $(TARGET)
