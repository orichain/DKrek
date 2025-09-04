all:
	gcc -c -Wall -O3 -mrdseed main.c -I/home/dhani/DATA/cuda/include -L/home/dhani/DATA/cuda/lib64 -o main.o
	nvcc -c -O3 --use_fast_math -gencode=arch=compute_75,code=sm_75 kernel.cu -o kernel.o
	nvcc -O3 --use_fast_math -gencode=arch=compute_75,code=sm_75 main.o kernel.o -lcudadevrt -o DKrek
	@rm -rf *.o

clean:
	@rm -rf *.o
