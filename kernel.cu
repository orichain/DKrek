#include <stdio.h>
#include <stdint.h>
#include "kernel.h"
#include "GPUSECP256K1.cuh"
#include "GPUHash.cuh"

#define SetD_Rmd(a,b) a[0] = b[0];a[1] = b[1];a[2] = b[2];a[3] = b[3];a[4] = b[4];a[5] = b[5];a[6] = b[6];a[7] = b[7];a[8] = b[8];a[9] = b[9];a[10] = b[10];a[11] = b[11];a[12] = b[12];a[13] = b[13];a[14] = b[14];a[15] = b[15];a[16] = b[16];a[17] = b[17];a[18] = b[18];a[19] = b[19];
#define IsSameD_Rmd(a,b) ((a[0] == b[0]) &&(a[1] == b[1]) &&(a[2] == b[2]) &&(a[3] == b[3]) &&(a[4] == b[4]) &&(a[5] == b[5]) &&(a[6] == b[6]) &&(a[7] == b[7]) &&(a[8] == b[8]) &&(a[9] == b[9]) &&(a[10] == b[10]) &&(a[11] == b[11]) &&(a[12] == b[12]) &&(a[13] == b[13]) &&(a[14] == b[14]) &&(a[15] == b[15]) &&(a[16] == b[16]) &&(a[17] == b[17]) &&(a[18] == b[18]) &&(a[19] == b[19]))

__device__ static void __noinline__ _ReduceAndRmdAndCompare(
	D_Point *publicKey, 
	D_Int256 *subp,
	D_Int *inverse,
	D_HashRmd *rmd160,
	uint32_t *indexKeyFound,
	D_HashRmd *S,
	uint32_t COUNT
) {
	D_Int _s;
	D_Int _p;
	D_Int dy;
	D_Int dx;
	D_Int dxInverse;
	D_HashRmd tmpRmd;
	D_Int X;
	D_Int Y;
	
	_ModInv(inverse);
#pragma unroll
	for (uint32_t yloop=(COUNT-1);yloop>1;yloop--) {
		_ModSub(&dy,&((GTableAdds[yloop-1]).Y),&((*publicKey).Y));
		_ModSub(&dx,&((GTableAdds[yloop-1]).X),&((*publicKey).X));
		_ModMulK1(&dxInverse, &(subp[yloop-2]), inverse);
		_ModMulK1(inverse, &dx);
		_ModMulK1(&_s,&dy,&dxInverse);
		_ModSquareK1(&_p,&_s);
		_ModSub(&X,&_p,&((*publicKey).X));
		_ModSub(&X,&((GTableAdds[yloop-1]).X));
		_ModSub(&Y,&((GTableAdds[yloop-1]).X),&X);
		_ModMulK1(&Y,&_s);
		_ModSub(&Y,&((GTableAdds[yloop-1]).Y));
        _GetHash160Comp(X.bitsu64, ((Y.bitsu32[0] & 1) == 1), tmpRmd);
        if (IsSameD_Rmd(tmpRmd, (*S))) {
			(*indexKeyFound) = yloop;
			SetD_Rmd((*rmd160), tmpRmd);
			break;
		} else {
			if (yloop == ((COUNT-1)-MINUSPRINTING)) {
				SetD_Rmd((*rmd160), tmpRmd);
			}
		}
    }
    if ((*indexKeyFound) == COUNT) {
		_ModSub(&dy,&((GTableAdds[0]).Y),&((*publicKey).Y));
		_ModMulK1(&_s,&dy,inverse);
		_ModSquareK1(&_p,&_s);
		_ModSub(&X,&_p,&((*publicKey).X));
		_ModSub(&X,&((GTableAdds[0]).X));
		_ModSub(&Y,&((GTableAdds[0]).X),&X);
		_ModMulK1(&Y,&_s);
		_ModSub(&Y,&((GTableAdds[0]).Y));
		_GetHash160Comp(X.bitsu64, ((Y.bitsu32[0] & 1) == 1), tmpRmd);
		if (IsSameD_Rmd(tmpRmd, (*S))) {
			(*indexKeyFound) = 1;
			SetD_Rmd((*rmd160), tmpRmd);
		} else {
			_GetHash160Comp(((*publicKey).X).bitsu64, ((((*publicKey).Y).bitsu32[0] & 1) == 1), tmpRmd);
			if (IsSameD_Rmd(tmpRmd, (*S))) {
				(*indexKeyFound) = 0;
				SetD_Rmd((*rmd160), tmpRmd);
			}
		}
	}
}

__global__ void _DKrek_Kernel(
	D_Int *dvc_privateKeyStart,
	D_Point *dvc_publicKeyStart,
	D_HashRmd *dvc_rmd160ToFind,
	D_Result *dvc_result,
	uint64_t size
) {
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	D_KeyPair dvc_keyPair;
	
	if (tid < size) {
		dvc_keyPair.indexKeyFound = GROUPCOUNT;
		_Add(&(dvc_keyPair.privateKey), dvc_privateKeyStart, tid*GROUPCOUNT);
		_ComputePublicKey(
			&(dvc_keyPair.privateKey),
			dvc_publicKeyStart,
			&(dvc_keyPair.publicKey), 
			dvc_keyPair.subp,
			&(dvc_keyPair.inverse),
			GROUPCOUNT
		);
		_ReduceAndRmdAndCompare(
			&(dvc_keyPair.publicKey), 
			dvc_keyPair.subp,
			&(dvc_keyPair.inverse),
			&(dvc_keyPair.rmd),
			&(dvc_keyPair.indexKeyFound), 
			dvc_rmd160ToFind,
			GROUPCOUNT
		);
		if (dvc_keyPair.indexKeyFound != GROUPCOUNT) {
			(*dvc_result).keyFound = 0x01;
			_SetD_Int(&((*dvc_result).privateKey), &(dvc_keyPair.privateKey));
			_Add(&((*dvc_result).privateKey), &((*dvc_result).privateKey), dvc_keyPair.indexKeyFound);
			SetD_Rmd((*dvc_result).rmd160, dvc_keyPair.rmd);
		}
		if ((*dvc_result).keyFound == 0x00) {
			if (tid == (size-1)) {
				(*dvc_result).keyFound = 0x00;
				_SetD_Int(&((*dvc_result).privateKey), &(dvc_keyPair.privateKey));
				_Add(&((*dvc_result).privateKey), &((*dvc_result).privateKey), ((GROUPCOUNT-1)-MINUSPRINTING));
				SetD_Rmd((*dvc_result).rmd160, dvc_keyPair.rmd);
			}
		}
	}
}

void _Launch_DKrek_Kernel(
	D_Int *privateKeyStart,
	D_Point *publicKeyStart,
	D_HashRmd *rmd160ToFind,	
	D_Result *res,
	uint64_t size
) {
	D_Int *dvc_privateKeyStart;
	D_Point *dvc_publicKeyStart;
	D_HashRmd *dvc_rmd160ToFind;
	D_Result *dvc_result;
	
	cudaMalloc((void**)&dvc_privateKeyStart, sizeof(D_Int));
	cudaMalloc((void**)&dvc_publicKeyStart, sizeof(D_Point));
	cudaMalloc((void**)&dvc_rmd160ToFind, sizeof(D_HashRmd));
	cudaMalloc((void**)&dvc_result, sizeof(D_Result));
	cudaMemcpy(dvc_privateKeyStart, privateKeyStart, sizeof(D_Int), cudaMemcpyHostToDevice);
	cudaMemcpy(dvc_publicKeyStart, publicKeyStart, sizeof(D_Point), cudaMemcpyHostToDevice);
	cudaMemcpy(dvc_rmd160ToFind, rmd160ToFind, sizeof(D_HashRmd), cudaMemcpyHostToDevice);
	uint64_t blockSize = 512;
	uint64_t numBlocks = (size + blockSize - 1) / blockSize;
	_DKrek_Kernel<<<numBlocks, blockSize>>>(
		dvc_privateKeyStart,
		dvc_publicKeyStart,
		dvc_rmd160ToFind,
		dvc_result,
		size
	);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA Error: %s\n", cudaGetErrorString(err));       
	}
	cudaDeviceSynchronize();
	cudaMemcpy(res, dvc_result, sizeof(D_Result), cudaMemcpyDeviceToHost);
	cudaFree(dvc_privateKeyStart);
	cudaFree(dvc_publicKeyStart);
	cudaFree(dvc_rmd160ToFind);
 	cudaFree(dvc_result);
}
