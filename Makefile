all:
	gcc -c -Wall -O3 base58.c -o base58.o
	gcc -c -Wall -O3 sha256.c -o sha256.o
	gcc -c -Wall -O3 -mrdseed main.c -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -o main.o
	/usr/local/cuda/bin/nvcc -c -O3 --use_fast_math -gencode=arch=compute_75,code=sm_75 kernel.cu -o kernel.o
	/usr/local/cuda/bin/nvcc -O3 --use_fast_math -gencode=arch=compute_75,code=sm_75 main.o base58.o sha256.o kernel.o -lcudadevrt -lsqlite3 -o DKrek
	@rm -rf *.o

clean:
	@rm -rf *.o
