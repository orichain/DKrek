CC       := gcc
NVCC     := /usr/local/cuda/bin/nvcc
CFLAGS   := -Wall -O3
GENCODES := -gencode arch=compute_75,code=sm_75 \
            -gencode arch=compute_80,code=sm_80 \
            -gencode arch=compute_86,code=sm_86 \
            -gencode arch=compute_89,code=sm_89 \
            -gencode arch=compute_90,code=sm_90 \
            -gencode arch=compute_90,code=compute_90
NVFLAGS  := -O3 --use_fast_math -Xptxas -v,-O3 $(GENCODES)

CUDA_INC := -I/usr/local/cuda/include
CUDA_LIB := -L/usr/local/cuda/lib64 -lcudart -lcudadevrt

TARGET   := DKrek
OBJS     := base58.o sha256.o main.o kernel.o

all: $(TARGET)

base58.o: base58.c
	$(CC) $(CFLAGS) -c $< -o $@

sha256.o: sha256.c
	$(CC) $(CFLAGS) -c $< -o $@

main.o: main.c
	$(CC) $(CFLAGS) -mrdseed $(CUDA_INC) -c $< -o $@

kernel.o: kernel.cu GPUSECP256K1.cuh GPUHash.cuh
	$(NVCC) $(NVFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(NVCC) $(NVFLAGS) $(OBJS) $(CUDA_LIB) -lsqlite3 -o $@

clean:
	rm -f *.o $(TARGET)

.PHONY: all clean
