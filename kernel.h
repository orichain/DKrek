#ifndef KERNEL_H
#define KERNEL_H

#include <stdint.h>
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

const uint8_t permutationsx[24][4] = {
	{0,1,2,3},
	{1,0,2,3},
	{2,0,1,3},
	{0,2,1,3},
	{1,2,0,3},
	{2,1,0,3},
	{2,1,3,0},
	{1,2,3,0},
	{3,2,1,0},
	{2,3,1,0},
	{1,3,2,0},
	{3,1,2,0},
	{3,0,2,1},
	{0,3,2,1},
	{2,3,0,1},
	{3,2,0,1},
	{0,2,3,1},
	{2,0,3,1},
	{1,0,3,2},
	{0,1,3,2},
	{3,1,0,2},
	{1,3,0,2},
	{0,3,1,2},
	{3,0,1,2},
};

void _Launch_DKrek_Kernel(
	D_Int *privateKeyStart,
	D_Point *publicKeyStart,
	D_HashRmd *rmd160ToFind,
	D_Result *res,
	uint64_t size
);

#ifdef __cplusplus
}
#endif

#endif
