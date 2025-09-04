#ifndef GPUSECP256K1_H
#define GPUSECP256K1_H

#include "const.cuh"

#define SWAP_ADD(x,y) x+=y;y-=x;
#define SWAP_SUB(x,y) x-=y;y+=x;
#define IS_EVEN(x) ((x&1)==0)
#define IS_ZERO(x) ((x.bitsu64[0] == 0ULL)&&(x.bitsu64[1] == 0ULL)&&(x.bitsu64[2] == 0ULL)&&(x.bitsu64[3] == 0ULL)&&(x.bitsu64[4] == 0ULL))
#define IS_ONE(x) ((x.bitsu64[0] == 1ULL)&&(x.bitsu64[1] == 0ULL)&&(x.bitsu64[2] == 0ULL)&&(x.bitsu64[3] == 0ULL)&&(x.bitsu64[4] == 0ULL))
#define IS_NEGATIVE(x) (x.bits64[4] < 0LL)
#define IS_POSITIVE(x) (x.bits64[4] >= 0LL)

__device__ static void __forceinline__ _SetD_Int(D_Int *a, D_Int *b) {
	(*a).bitsu64[0] = (*b).bitsu64[0];
	(*a).bitsu64[1] = (*b).bitsu64[1];
	(*a).bitsu64[2] = (*b).bitsu64[2];
	(*a).bitsu64[3] = (*b).bitsu64[3];
	(*a).bitsu64[4] = (*b).bitsu64[4];
}

__device__ static void __forceinline__ _SetD_Int(D_Int256 *a, D_Int *b) {
	(*a).bitsu64[0] = (*b).bitsu64[0];
	(*a).bitsu64[1] = (*b).bitsu64[1];
	(*a).bitsu64[2] = (*b).bitsu64[2];
	(*a).bitsu64[3] = (*b).bitsu64[3];
}

__device__ static void __forceinline__ _SetD_Int(D_Int *a, D_Int256 *b) {
	(*a).bitsu64[0] = (*b).bitsu64[0];
	(*a).bitsu64[1] = (*b).bitsu64[1];
	(*a).bitsu64[2] = (*b).bitsu64[2];
	(*a).bitsu64[3] = (*b).bitsu64[3];
	(*a).bitsu64[4] = 0ULL;
}

__device__ static void __forceinline__ _SetD_Int_One(D_Int *a) {
	(*a).bitsu64[0] = 1ULL;
	(*a).bitsu64[1] = 0ULL;
	(*a).bitsu64[2] = 0ULL;
	(*a).bitsu64[3] = 0ULL;
	(*a).bitsu64[4] = 0ULL;
}

__device__ static void __forceinline__ _ClearD_Int(D_Int *a) {
	(*a).bitsu64[0] = 0ULL;
	(*a).bitsu64[1] = 0ULL;
	(*a).bitsu64[2] = 0ULL;
	(*a).bitsu64[3] = 0ULL;
	(*a).bitsu64[4] = 0ULL;
}

__device__ static void __forceinline__ _SetD_Point(D_Point *a, D_Point *b) {
	_SetD_Int(&((*a).X), &((*b).X));
	_SetD_Int(&((*a).Y), &((*b).Y));
	_SetD_Int(&((*a).Z), &((*b).Z));
}

__device__ static void __forceinline__ _ClearD_Point(D_Point *a) {
	_ClearD_Int(&((*a).X));
	_ClearD_Int(&((*a).Y));
	_ClearD_Int(&((*a).Z));
}

__device__ static void __forceinline__ _shiftR(uint32_t n, D_Int *d) {
	asm("{\n\t"
		".reg .u32 rn;\n\t"
		".reg .u64 r1, r2;\n\t"
		"sub.u32 rn, 64, %9;\n\t"
		
		"shl.b64 r1, %5, rn;\n\t"
		"shr.b64 r2, %4, %9;\n\t"		
		"or.b64  %0, r1, r2;\n\t"
		"shl.b64 r1, %6, rn;\n\t"
		"shr.b64 r2, %5, %9;\n\t"		
		"or.b64  %1, r1, r2;\n\t"
		"shl.b64 r1, %7, rn;\n\t"
		"shr.b64 r2, %6, %9;\n\t"		
		"or.b64  %2, r1, r2;\n\t"
		"shl.b64 r1, %8, rn;\n\t"
		"shr.b64 r2, %7, %9;\n\t"		
		"or.b64  %3, r1, r2;\n\t"
		"}\n\t"
		: "=l"((*d).bitsu64[0]), "=l"((*d).bitsu64[1]), "=l"((*d).bitsu64[2]), "=l"((*d).bitsu64[3])
		: "l"((*d).bitsu64[0]), "l"((*d).bitsu64[1]), "l"((*d).bitsu64[2]), "l"((*d).bitsu64[3]), "l"((*d).bitsu64[4]), "r"(n)
		: "memory"
	);
	(*d).bitsu64[4] = (*d).bits64[4] >> n;
}

__device__ static void __forceinline__ _imm_umul(uint64_t *x, uint64_t y, uint64_t *dst) {
	asm(
		"{\n\t"
		".reg .u64 c1;\n\t"
		"mul.lo.u64 %0, %5, %9;\r\n"
		"mul.hi.u64 c1, %5, %9;\r\n"
		"mad.lo.cc.u64 %1, %6, %9, c1;\r\n"
		"mul.hi.u64 c1, %6, %9;\r\n"
		"madc.lo.cc.u64 %2, %7, %9, c1;\r\n"
		"mul.hi.u64 c1, %7, %9;\r\n"
		"madc.lo.cc.u64 %3, %8, %9, c1;\r\n"
		"mul.hi.u64 c1, %8, %9;\r\n"
		"madc.lo.u64 %4, 0, %9, c1;\r\n"
		"}\n\t"
		: "=l"(dst[0]), "=l"(dst[1]), "=l"(dst[2]), "=l"(dst[3]), "=l"(dst[4])
		: "l"(x[0]), "l"(x[1]), "l"(x[2]), "l"(x[3]), "l"(y) 
		: "memory"
	);
}

__device__ static void __forceinline__ _imm_mul(uint64_t *x, uint64_t y, uint64_t *dst) {
	asm(
		"{\n\t"
		".reg .u64 c1;\n\t"
		"mul.lo.u64 %0, %5, %10;\r\n"
		"mul.hi.u64 c1, %5, %10;\r\n"		
		"mad.lo.cc.u64 %1, %6, %10, c1;\r\n"		
		"mul.hi.u64 c1, %6, %10;\r\n"
		"madc.lo.cc.u64 %2, %7, %10, c1;\r\n"		
		"mul.hi.u64 c1, %7, %10;\r\n"		
		"madc.lo.cc.u64 %3, %8, %10, c1;\r\n"
		"mul.hi.u64 c1, %8, %10;\r\n"
		"mad.lo.u64 %4, %9, %10, c1;\r\n"
		"}\n\t"
		: "=l"(dst[0]), "=l"(dst[1]), "=l"(dst[2]), "=l"(dst[3]), "=l"(dst[4])
		: "l"(x[0]), "l"(x[1]), "l"(x[2]), "l"(x[3]), "l"(x[4]), "l"(y) 
		: "memory"
	);
}

__device__ static void __forceinline__ _Add(D_Int *RES, D_Int *a, uint64_t b) {
	asm(
		"{\n\t"
		"add.cc.u64 %0, %5, %10;\r\n"		
		"addc.cc.u64 %1, %6, 0;\r\n"		
		"addc.cc.u64 %2, %7, 0;\r\n"		
		"addc.cc.u64 %3, %8, 0;\r\n"		
		"addc.u64 %4, %9, 0;\r\n"
		"}\n\t"
		: "=l"((*RES).bitsu64[0]), "=l"((*RES).bitsu64[1]), "=l"((*RES).bitsu64[2]), "=l"((*RES).bitsu64[3]), "=l"((*RES).bitsu64[4])
		: "l"((*a).bitsu64[0]), "l"((*a).bitsu64[1]), "l"((*a).bitsu64[2]), "l"((*a).bitsu64[3]), "l"((*a).bitsu64[4]), "l"(b) 
		: "memory"
	);
}

__device__ static void __forceinline__ _Add(D_Int *RES, D_Int *a) {
	asm(
		"{\n\t"
		"add.cc.u64 %0, %5, %10;\r\n"		
		"addc.cc.u64 %1, %6, %11;\r\n"		
		"addc.cc.u64 %2, %7, %12;\r\n"		
		"addc.cc.u64 %3, %8, %13;\r\n"		
		"addc.u64 %4, %9, %14;\r\n"
		"}\n\t"
		: "=l"((*RES).bitsu64[0]), "=l"((*RES).bitsu64[1]), "=l"((*RES).bitsu64[2]), "=l"((*RES).bitsu64[3]), "=l"((*RES).bitsu64[4])
		: "l"((*RES).bitsu64[0]), "l"((*RES).bitsu64[1]), "l"((*RES).bitsu64[2]), "l"((*RES).bitsu64[3]), "l"((*RES).bitsu64[4]), "l"((*a).bitsu64[0]), "l"((*a).bitsu64[1]), "l"((*a).bitsu64[2]), "l"((*a).bitsu64[3]), "l"((*a).bitsu64[4])
		: "memory"
	);
}

__device__ static void __forceinline__ _Add(D_Int *RES, D_Int *a, D_Int *b) {
	asm(
		"{\n\t"
		"add.cc.u64 %0, %5, %10;\r\n"		
		"addc.cc.u64 %1, %6, %11;\r\n"		
		"addc.cc.u64 %2, %7, %12;\r\n"		
		"addc.cc.u64 %3, %8, %13;\r\n"		
		"addc.u64 %4, %9, %14;\r\n"
		"}\n\t"
		: "=l"((*RES).bitsu64[0]), "=l"((*RES).bitsu64[1]), "=l"((*RES).bitsu64[2]), "=l"((*RES).bitsu64[3]), "=l"((*RES).bitsu64[4])
		: "l"((*a).bitsu64[0]), "l"((*a).bitsu64[1]), "l"((*a).bitsu64[2]), "l"((*a).bitsu64[3]), "l"((*a).bitsu64[4]), "l"((*b).bitsu64[0]), "l"((*b).bitsu64[1]), "l"((*b).bitsu64[2]), "l"((*b).bitsu64[3]), "l"((*b).bitsu64[4])
		: "memory"
	);
}

__device__ static uint64_t __forceinline__ _AddC(D_Int *RES, D_Int *a) {
	uint64_t c = 0;
	
	asm(
		"{\n\t"
		"add.cc.u64 %0, %6, %11;\r\n"		
		"addc.cc.u64 %1, %7, %12;\r\n"		
		"addc.cc.u64 %2, %8, %13;\r\n"		
		"addc.cc.u64 %3, %9, %14;\r\n"		
		"addc.cc.u64 %4, %10, %15;\r\n"
		"addc.u64 %5, 0, 0;\r\n"
		"}\n\t"
		: "=l"((*RES).bitsu64[0]), "=l"((*RES).bitsu64[1]), "=l"((*RES).bitsu64[2]), "=l"((*RES).bitsu64[3]), "=l"((*RES).bitsu64[4]), "=l"(c)
		: "l"((*RES).bitsu64[0]), "l"((*RES).bitsu64[1]), "l"((*RES).bitsu64[2]), "l"((*RES).bitsu64[3]), "l"((*RES).bitsu64[4]), "l"((*a).bitsu64[0]), "l"((*a).bitsu64[1]), "l"((*a).bitsu64[2]), "l"((*a).bitsu64[3]), "l"((*a).bitsu64[4])
		: "memory"
	);
	return c;
}

__device__ static void __forceinline__ _AddAndShift(D_Int *RES, D_Int *a, uint64_t cH) {
	asm(
		"{\n\t"
		"add.cc.u64 %0, %5, %10;\r\n"
		"addc.cc.u64 %0, %6, %11;\r\n"
		"addc.cc.u64 %1, %7, %12;\r\n"
		"addc.cc.u64 %2, %8, %13;\r\n"
		"addc.cc.u64 %3, %9, %14;\r\n"
		"addc.u64 %4, 0, %15;\r\n"
		"}\n\t"
		: "=l"((*RES).bitsu64[0]), "=l"((*RES).bitsu64[1]), "=l"((*RES).bitsu64[2]), "=l"((*RES).bitsu64[3]), "=l"((*RES).bitsu64[4])
		: "l"((*RES).bitsu64[0]), "l"((*RES).bitsu64[1]), "l"((*RES).bitsu64[2]), "l"((*RES).bitsu64[3]), "l"((*RES).bitsu64[4]), "l"((*a).bitsu64[0]), "l"((*a).bitsu64[1]), "l"((*a).bitsu64[2]), "l"((*a).bitsu64[3]), "l"((*a).bitsu64[4]), "l"(cH)
		: "memory"
	);
}

__device__ static void __forceinline__ _Sub(D_Int *RES, D_Int *a, D_Int *b) {
	asm(
		"{\n\t"
		"sub.cc.u64 %0, %5, %10;\r\n"		
		"subc.cc.u64 %1, %6, %11;\r\n"		
		"subc.cc.u64 %2, %7, %12;\r\n"		
		"subc.cc.u64 %3, %8, %13;\r\n"		
		"subc.u64 %4, %9, %14;\r\n"
		"}\n\t"
		: "=l"((*RES).bitsu64[0]), "=l"((*RES).bitsu64[1]), "=l"((*RES).bitsu64[2]), "=l"((*RES).bitsu64[3]), "=l"((*RES).bitsu64[4])
		: "l"((*a).bitsu64[0]), "l"((*a).bitsu64[1]), "l"((*a).bitsu64[2]), "l"((*a).bitsu64[3]), "l"((*a).bitsu64[4]), "l"((*b).bitsu64[0]), "l"((*b).bitsu64[1]), "l"((*b).bitsu64[2]), "l"((*b).bitsu64[3]), "l"((*b).bitsu64[4])
		: "memory"
	);
}

__device__ static void __forceinline__ _Sub(D_Int *RES, D_Int *a) {
	asm(
		"{\n\t"
		"sub.cc.u64 %0, %5, %10;\r\n"		
		"subc.cc.u64 %1, %6, %11;\r\n"		
		"subc.cc.u64 %2, %7, %12;\r\n"		
		"subc.cc.u64 %3, %8, %13;\r\n"		
		"subc.u64 %4, %9, %14;\r\n"
		"}\n\t"
		: "=l"((*RES).bitsu64[0]), "=l"((*RES).bitsu64[1]), "=l"((*RES).bitsu64[2]), "=l"((*RES).bitsu64[3]), "=l"((*RES).bitsu64[4])
		: "l"((*RES).bitsu64[0]), "l"((*RES).bitsu64[1]), "l"((*RES).bitsu64[2]), "l"((*RES).bitsu64[3]), "l"((*RES).bitsu64[4]), "l"((*a).bitsu64[0]), "l"((*a).bitsu64[1]), "l"((*a).bitsu64[2]), "l"((*a).bitsu64[3]), "l"((*a).bitsu64[4])
		: "memory"
	);
}

__device__ static void __forceinline__ _Sub(D_Int *RES, uint64_t a) {
	asm(
		"{\n\t"
		"sub.cc.u64 %0, %5, %10;\r\n"		
		"subc.cc.u64 %1, %6, 0;\r\n"		
		"subc.cc.u64 %2, %7, 0;\r\n"		
		"subc.cc.u64 %3, %8, 0;\r\n"		
		"subc.u64 %4, %9, 0;\r\n"
		"}\n\t"
		: "=l"((*RES).bitsu64[0]), "=l"((*RES).bitsu64[1]), "=l"((*RES).bitsu64[2]), "=l"((*RES).bitsu64[3]), "=l"((*RES).bitsu64[4])
		: "l"((*RES).bitsu64[0]), "l"((*RES).bitsu64[1]), "l"((*RES).bitsu64[2]), "l"((*RES).bitsu64[3]), "l"((*RES).bitsu64[4]), "l"(a)
		: "memory"
	);
}

__device__ static void __forceinline__ _Neg(D_Int *RES) {
	asm(
		"{\n\t"
		"sub.cc.u64 %0, 0, %5;\r\n"		
		"subc.cc.u64 %1, 0, %6;\r\n"		
		"subc.cc.u64 %2, 0, %7;\r\n"		
		"subc.cc.u64 %3, 0, %8;\r\n"		
		"subc.u64 %4, 0, %9;\r\n"
		"}\n\t"
		: "=l"((*RES).bitsu64[0]), "=l"((*RES).bitsu64[1]), "=l"((*RES).bitsu64[2]), "=l"((*RES).bitsu64[3]), "=l"((*RES).bitsu64[4])
		: "l"((*RES).bitsu64[0]), "l"((*RES).bitsu64[1]), "l"((*RES).bitsu64[2]), "l"((*RES).bitsu64[3]), "l"((*RES).bitsu64[4])
		: "memory"
	);
}

__device__ static void __forceinline__ _Mult(D_Int *RES, D_Int *a, uint64_t b) {
	_imm_mul(a->bitsu64, b, RES->bitsu64);
}

__device__ static void __forceinline__ _IMult(D_Int *RES, D_Int *a, int64_t b) {
	_SetD_Int(RES, a);

	if (b < 0LL) {
		_Neg(RES);
		b = -b;
	}
	_imm_mul(RES->bitsu64, b, RES->bitsu64);
}

__device__ static void __forceinline__ _ModMulK1(D_Int *RES, D_Int *a, D_Int *b) {
	uint64_t ah, al, c;
	uint64_t t[5];
	uint64_t r512[8];
	r512[5] = 0;
	r512[6] = 0;
	r512[7] = 0;

	_imm_umul(a->bitsu64, b->bitsu64[0], r512);
	_imm_umul(a->bitsu64, b->bitsu64[1], t);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %5, %10;\r\n"		
		"addc.cc.u64 %1, %6, %11;\r\n"		
		"addc.cc.u64 %2, %7, %12;\r\n"		
		"addc.cc.u64 %3, %8, %13;\r\n"		
		"addc.u64 %4, %9, %14;\r\n"
		"}\n\t"
		: "=l"(r512[1]), "=l"(r512[2]), "=l"(r512[3]), "=l"(r512[4]), "=l"(r512[5])
		: "l"(r512[1]), "l"(r512[2]), "l"(r512[3]), "l"(r512[4]), "l"(r512[5]), "l"(t[0]), "l"(t[1]), "l"(t[2]), "l"(t[3]), "l"(t[4])
		: "memory"
	);
	_imm_umul(a->bitsu64, b->bitsu64[2], t);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %5, %10;\r\n"		
		"addc.cc.u64 %1, %6, %11;\r\n"		
		"addc.cc.u64 %2, %7, %12;\r\n"		
		"addc.cc.u64 %3, %8, %13;\r\n"		
		"addc.u64 %4, %9, %14;\r\n"
		"}\n\t"
		: "=l"(r512[2]), "=l"(r512[3]), "=l"(r512[4]), "=l"(r512[5]), "=l"(r512[6])
		: "l"(r512[2]), "l"(r512[3]), "l"(r512[4]), "l"(r512[5]), "l"(r512[6]), "l"(t[0]), "l"(t[1]), "l"(t[2]), "l"(t[3]), "l"(t[4])
		: "memory"
	);
	_imm_umul(a->bitsu64, b->bitsu64[3], t);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %5, %10;\r\n"		
		"addc.cc.u64 %1, %6, %11;\r\n"		
		"addc.cc.u64 %2, %7, %12;\r\n"		
		"addc.cc.u64 %3, %8, %13;\r\n"		
		"addc.u64 %4, %9, %14;\r\n"
		"}\n\t"
		: "=l"(r512[3]), "=l"(r512[4]), "=l"(r512[5]), "=l"(r512[6]), "=l"(r512[7])
		: "l"(r512[3]), "l"(r512[4]), "l"(r512[5]), "l"(r512[6]), "l"(r512[7]), "l"(t[0]), "l"(t[1]), "l"(t[2]), "l"(t[3]), "l"(t[4])
		: "memory"
	);
	// Reduce from 512 to 320 
	_imm_umul(r512 + 4, 0x1000003D1ULL, t);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %5, %9;\r\n"		
		"addc.cc.u64 %1, %6, %10;\r\n"		
		"addc.cc.u64 %2, %7, %11;\r\n"		
		"addc.cc.u64 %3, %8, %12;\r\n"		
		"addc.u64 %4, 0, 0;\r\n"
		"}\n\t"
		: "=l"(r512[0]), "=l"(r512[1]), "=l"(r512[2]), "=l"(r512[3]), "=l"(c)
		: "l"(r512[0]), "l"(r512[1]), "l"(r512[2]), "l"(r512[3]), "l"(t[0]), "l"(t[1]), "l"(t[2]), "l"(t[3])
		: "memory"
	);
	// Reduce from 320 to 256 
	// No overflow possible here t[4]+c<=0x1000003D1ULL
	asm(
		"{\n\t"
		"mul.lo.u64 %0, %2, %3;\r\n"
		"mul.hi.u64 %1, %2, %3;\r\n"
		"}\n\t"
		: "=l"(al), "=l"(ah)
		: "l"(t[4] + c), "l"(0x1000003D1ULL)
		: "memory"
	);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %4, %8;\r\n"
		"addc.cc.u64 %1, %5, %9;\r\n"
		"addc.cc.u64 %2, %6, 0;\r\n"
		"addc.u64 %3, %7, 0;\r\n"
		"}\n\t"
		: "=l"((*RES).bitsu64[0]), "=l"((*RES).bitsu64[1]), "=l"((*RES).bitsu64[2]), "=l"((*RES).bitsu64[3])
		: "l"(r512[0]), "l"(r512[1]), "l"(r512[2]), "l"(r512[3]), "l"(al), "l"(ah)
		: "memory"
	);
	// Probability of carry here or that this>P is very very unlikely
	RES->bitsu64[4] = 0ULL; 
}

__device__ static void __forceinline__ _ModMulK1(D_Int *RES, D_Int256 *a, D_Int *b) {
	uint64_t ah, al, c;
	uint64_t t[5];
	uint64_t r512[8];
	r512[5] = 0;
	r512[6] = 0;
	r512[7] = 0;

	_imm_umul(a->bitsu64, b->bitsu64[0], r512);
	_imm_umul(a->bitsu64, b->bitsu64[1], t);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %5, %10;\r\n"		
		"addc.cc.u64 %1, %6, %11;\r\n"		
		"addc.cc.u64 %2, %7, %12;\r\n"		
		"addc.cc.u64 %3, %8, %13;\r\n"		
		"addc.u64 %4, %9, %14;\r\n"
		"}\n\t"
		: "=l"(r512[1]), "=l"(r512[2]), "=l"(r512[3]), "=l"(r512[4]), "=l"(r512[5])
		: "l"(r512[1]), "l"(r512[2]), "l"(r512[3]), "l"(r512[4]), "l"(r512[5]), "l"(t[0]), "l"(t[1]), "l"(t[2]), "l"(t[3]), "l"(t[4])
		: "memory"
	);
	_imm_umul(a->bitsu64, b->bitsu64[2], t);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %5, %10;\r\n"		
		"addc.cc.u64 %1, %6, %11;\r\n"		
		"addc.cc.u64 %2, %7, %12;\r\n"		
		"addc.cc.u64 %3, %8, %13;\r\n"		
		"addc.u64 %4, %9, %14;\r\n"
		"}\n\t"
		: "=l"(r512[2]), "=l"(r512[3]), "=l"(r512[4]), "=l"(r512[5]), "=l"(r512[6])
		: "l"(r512[2]), "l"(r512[3]), "l"(r512[4]), "l"(r512[5]), "l"(r512[6]), "l"(t[0]), "l"(t[1]), "l"(t[2]), "l"(t[3]), "l"(t[4])
		: "memory"
	);
	_imm_umul(a->bitsu64, b->bitsu64[3], t);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %5, %10;\r\n"		
		"addc.cc.u64 %1, %6, %11;\r\n"		
		"addc.cc.u64 %2, %7, %12;\r\n"		
		"addc.cc.u64 %3, %8, %13;\r\n"		
		"addc.u64 %4, %9, %14;\r\n"
		"}\n\t"
		: "=l"(r512[3]), "=l"(r512[4]), "=l"(r512[5]), "=l"(r512[6]), "=l"(r512[7])
		: "l"(r512[3]), "l"(r512[4]), "l"(r512[5]), "l"(r512[6]), "l"(r512[7]), "l"(t[0]), "l"(t[1]), "l"(t[2]), "l"(t[3]), "l"(t[4])
		: "memory"
	);
	// Reduce from 512 to 320 
	_imm_umul(r512 + 4, 0x1000003D1ULL, t);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %5, %9;\r\n"		
		"addc.cc.u64 %1, %6, %10;\r\n"		
		"addc.cc.u64 %2, %7, %11;\r\n"		
		"addc.cc.u64 %3, %8, %12;\r\n"		
		"addc.u64 %4, 0, 0;\r\n"
		"}\n\t"
		: "=l"(r512[0]), "=l"(r512[1]), "=l"(r512[2]), "=l"(r512[3]), "=l"(c)
		: "l"(r512[0]), "l"(r512[1]), "l"(r512[2]), "l"(r512[3]), "l"(t[0]), "l"(t[1]), "l"(t[2]), "l"(t[3])
		: "memory"
	);
	// Reduce from 320 to 256 
	// No overflow possible here t[4]+c<=0x1000003D1ULL
	asm(
		"{\n\t"
		"mul.lo.u64 %0, %2, %3;\r\n"
		"mul.hi.u64 %1, %2, %3;\r\n"
		"}\n\t"
		: "=l"(al), "=l"(ah)
		: "l"(t[4] + c), "l"(0x1000003D1ULL)
		: "memory"
	);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %4, %8;\r\n"
		"addc.cc.u64 %1, %5, %9;\r\n"
		"addc.cc.u64 %2, %6, 0;\r\n"
		"addc.u64 %3, %7, 0;\r\n"
		"}\n\t"
		: "=l"((*RES).bitsu64[0]), "=l"((*RES).bitsu64[1]), "=l"((*RES).bitsu64[2]), "=l"((*RES).bitsu64[3])
		: "l"(r512[0]), "l"(r512[1]), "l"(r512[2]), "l"(r512[3]), "l"(al), "l"(ah)
		: "memory"
	);
	// Probability of carry here or that this>P is very very unlikely
	RES->bitsu64[4] = 0ULL; 
}

__device__ static void __forceinline__ _ModMulK1(D_Int256 *RES, D_Int256 *a, D_Int *b) {
	uint64_t ah, al, c;
	uint64_t t[5];
	uint64_t r512[8];
	r512[5] = 0;
	r512[6] = 0;
	r512[7] = 0;

	_imm_umul(a->bitsu64, b->bitsu64[0], r512);
	_imm_umul(a->bitsu64, b->bitsu64[1], t);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %5, %10;\r\n"		
		"addc.cc.u64 %1, %6, %11;\r\n"		
		"addc.cc.u64 %2, %7, %12;\r\n"		
		"addc.cc.u64 %3, %8, %13;\r\n"		
		"addc.u64 %4, %9, %14;\r\n"
		"}\n\t"
		: "=l"(r512[1]), "=l"(r512[2]), "=l"(r512[3]), "=l"(r512[4]), "=l"(r512[5])
		: "l"(r512[1]), "l"(r512[2]), "l"(r512[3]), "l"(r512[4]), "l"(r512[5]), "l"(t[0]), "l"(t[1]), "l"(t[2]), "l"(t[3]), "l"(t[4])
		: "memory"
	);
	_imm_umul(a->bitsu64, b->bitsu64[2], t);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %5, %10;\r\n"		
		"addc.cc.u64 %1, %6, %11;\r\n"		
		"addc.cc.u64 %2, %7, %12;\r\n"		
		"addc.cc.u64 %3, %8, %13;\r\n"		
		"addc.u64 %4, %9, %14;\r\n"
		"}\n\t"
		: "=l"(r512[2]), "=l"(r512[3]), "=l"(r512[4]), "=l"(r512[5]), "=l"(r512[6])
		: "l"(r512[2]), "l"(r512[3]), "l"(r512[4]), "l"(r512[5]), "l"(r512[6]), "l"(t[0]), "l"(t[1]), "l"(t[2]), "l"(t[3]), "l"(t[4])
		: "memory"
	);
	_imm_umul(a->bitsu64, b->bitsu64[3], t);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %5, %10;\r\n"		
		"addc.cc.u64 %1, %6, %11;\r\n"		
		"addc.cc.u64 %2, %7, %12;\r\n"		
		"addc.cc.u64 %3, %8, %13;\r\n"		
		"addc.u64 %4, %9, %14;\r\n"
		"}\n\t"
		: "=l"(r512[3]), "=l"(r512[4]), "=l"(r512[5]), "=l"(r512[6]), "=l"(r512[7])
		: "l"(r512[3]), "l"(r512[4]), "l"(r512[5]), "l"(r512[6]), "l"(r512[7]), "l"(t[0]), "l"(t[1]), "l"(t[2]), "l"(t[3]), "l"(t[4])
		: "memory"
	);
	// Reduce from 512 to 320 
	_imm_umul(r512 + 4, 0x1000003D1ULL, t);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %5, %9;\r\n"		
		"addc.cc.u64 %1, %6, %10;\r\n"		
		"addc.cc.u64 %2, %7, %11;\r\n"		
		"addc.cc.u64 %3, %8, %12;\r\n"		
		"addc.u64 %4, 0, 0;\r\n"
		"}\n\t"
		: "=l"(r512[0]), "=l"(r512[1]), "=l"(r512[2]), "=l"(r512[3]), "=l"(c)
		: "l"(r512[0]), "l"(r512[1]), "l"(r512[2]), "l"(r512[3]), "l"(t[0]), "l"(t[1]), "l"(t[2]), "l"(t[3])
		: "memory"
	);
	// Reduce from 320 to 256 
	// No overflow possible here t[4]+c<=0x1000003D1ULL
	asm(
		"{\n\t"
		"mul.lo.u64 %0, %2, %3;\r\n"
		"mul.hi.u64 %1, %2, %3;\r\n"
		"}\n\t"
		: "=l"(al), "=l"(ah)
		: "l"(t[4] + c), "l"(0x1000003D1ULL)
		: "memory"
	);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %4, %8;\r\n"
		"addc.cc.u64 %1, %5, %9;\r\n"
		"addc.cc.u64 %2, %6, 0;\r\n"
		"addc.u64 %3, %7, 0;\r\n"
		"}\n\t"
		: "=l"((*RES).bitsu64[0]), "=l"((*RES).bitsu64[1]), "=l"((*RES).bitsu64[2]), "=l"((*RES).bitsu64[3])
		: "l"(r512[0]), "l"(r512[1]), "l"(r512[2]), "l"(r512[3]), "l"(al), "l"(ah)
		: "memory"
	);
}

__device__ static void __forceinline__ _ModMulK1(D_Int *RES, D_Int *a) {
	uint64_t ah, al, c;
	uint64_t t[5];
	uint64_t r512[8];
	r512[5] = 0;
	r512[6] = 0;
	r512[7] = 0;

	_imm_umul(RES->bitsu64, a->bitsu64[0], r512);
	_imm_umul(RES->bitsu64, a->bitsu64[1], t);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %5, %10;\r\n"		
		"addc.cc.u64 %1, %6, %11;\r\n"		
		"addc.cc.u64 %2, %7, %12;\r\n"		
		"addc.cc.u64 %3, %8, %13;\r\n"		
		"addc.u64 %4, %9, %14;\r\n"
		"}\n\t"
		: "=l"(r512[1]), "=l"(r512[2]), "=l"(r512[3]), "=l"(r512[4]), "=l"(r512[5])
		: "l"(r512[1]), "l"(r512[2]), "l"(r512[3]), "l"(r512[4]), "l"(r512[5]), "l"(t[0]), "l"(t[1]), "l"(t[2]), "l"(t[3]), "l"(t[4])
		: "memory"
	);
	_imm_umul(RES->bitsu64, a->bitsu64[2], t);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %5, %10;\r\n"		
		"addc.cc.u64 %1, %6, %11;\r\n"		
		"addc.cc.u64 %2, %7, %12;\r\n"		
		"addc.cc.u64 %3, %8, %13;\r\n"		
		"addc.u64 %4, %9, %14;\r\n"
		"}\n\t"
		: "=l"(r512[2]), "=l"(r512[3]), "=l"(r512[4]), "=l"(r512[5]), "=l"(r512[6])
		: "l"(r512[2]), "l"(r512[3]), "l"(r512[4]), "l"(r512[5]), "l"(r512[6]), "l"(t[0]), "l"(t[1]), "l"(t[2]), "l"(t[3]), "l"(t[4])
		: "memory"
	);
	_imm_umul(RES->bitsu64, a->bitsu64[3], t);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %5, %10;\r\n"		
		"addc.cc.u64 %1, %6, %11;\r\n"		
		"addc.cc.u64 %2, %7, %12;\r\n"		
		"addc.cc.u64 %3, %8, %13;\r\n"		
		"addc.u64 %4, %9, %14;\r\n"
		"}\n\t"
		: "=l"(r512[3]), "=l"(r512[4]), "=l"(r512[5]), "=l"(r512[6]), "=l"(r512[7])
		: "l"(r512[3]), "l"(r512[4]), "l"(r512[5]), "l"(r512[6]), "l"(r512[7]), "l"(t[0]), "l"(t[1]), "l"(t[2]), "l"(t[3]), "l"(t[4])
		: "memory"
	);
	// Reduce from 512 to 320 
	_imm_umul(r512 + 4, 0x1000003D1ULL, t);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %5, %9;\r\n"		
		"addc.cc.u64 %1, %6, %10;\r\n"		
		"addc.cc.u64 %2, %7, %11;\r\n"		
		"addc.cc.u64 %3, %8, %12;\r\n"		
		"addc.u64 %4, 0, 0;\r\n"
		"}\n\t"
		: "=l"(r512[0]), "=l"(r512[1]), "=l"(r512[2]), "=l"(r512[3]), "=l"(c)
		: "l"(r512[0]), "l"(r512[1]), "l"(r512[2]), "l"(r512[3]), "l"(t[0]), "l"(t[1]), "l"(t[2]), "l"(t[3])
		: "memory"
	);
	// Reduce from 320 to 256 
	// No overflow possible here t[4]+c<=0x1000003D1ULL
	asm(
		"{\n\t"
		"mul.lo.u64 %0, %2, %3;\r\n"
		"mul.hi.u64 %1, %2, %3;\r\n"
		"}\n\t"
		: "=l"(al), "=l"(ah)
		: "l"(t[4] + c), "l"(0x1000003D1ULL)
		: "memory"
	);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %4, %8;\r\n"
		"addc.cc.u64 %1, %5, %9;\r\n"
		"addc.cc.u64 %2, %6, 0;\r\n"
		"addc.u64 %3, %7, 0;\r\n"
		"}\n\t"
		: "=l"((*RES).bitsu64[0]), "=l"((*RES).bitsu64[1]), "=l"((*RES).bitsu64[2]), "=l"((*RES).bitsu64[3])
		: "l"(r512[0]), "l"(r512[1]), "l"(r512[2]), "l"(r512[3]), "l"(al), "l"(ah)
		: "memory"
	);
	// Probability of carry here or that this>P is very very unlikely
	RES->bitsu64[4] = 0ULL; 
}

__device__ static void __forceinline__ _ModMulK1(D_Int256 *RES, D_Int *a) {
	uint64_t ah, al, c;
	uint64_t t[5];
	uint64_t r512[8];
	r512[5] = 0;
	r512[6] = 0;
	r512[7] = 0;

	_imm_umul(RES->bitsu64, a->bitsu64[0], r512);
	_imm_umul(RES->bitsu64, a->bitsu64[1], t);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %5, %10;\r\n"		
		"addc.cc.u64 %1, %6, %11;\r\n"		
		"addc.cc.u64 %2, %7, %12;\r\n"		
		"addc.cc.u64 %3, %8, %13;\r\n"		
		"addc.u64 %4, %9, %14;\r\n"
		"}\n\t"
		: "=l"(r512[1]), "=l"(r512[2]), "=l"(r512[3]), "=l"(r512[4]), "=l"(r512[5])
		: "l"(r512[1]), "l"(r512[2]), "l"(r512[3]), "l"(r512[4]), "l"(r512[5]), "l"(t[0]), "l"(t[1]), "l"(t[2]), "l"(t[3]), "l"(t[4])
		: "memory"
	);
	_imm_umul(RES->bitsu64, a->bitsu64[2], t);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %5, %10;\r\n"		
		"addc.cc.u64 %1, %6, %11;\r\n"		
		"addc.cc.u64 %2, %7, %12;\r\n"		
		"addc.cc.u64 %3, %8, %13;\r\n"		
		"addc.u64 %4, %9, %14;\r\n"
		"}\n\t"
		: "=l"(r512[2]), "=l"(r512[3]), "=l"(r512[4]), "=l"(r512[5]), "=l"(r512[6])
		: "l"(r512[2]), "l"(r512[3]), "l"(r512[4]), "l"(r512[5]), "l"(r512[6]), "l"(t[0]), "l"(t[1]), "l"(t[2]), "l"(t[3]), "l"(t[4])
		: "memory"
	);
	_imm_umul(RES->bitsu64, a->bitsu64[3], t);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %5, %10;\r\n"		
		"addc.cc.u64 %1, %6, %11;\r\n"		
		"addc.cc.u64 %2, %7, %12;\r\n"		
		"addc.cc.u64 %3, %8, %13;\r\n"		
		"addc.u64 %4, %9, %14;\r\n"
		"}\n\t"
		: "=l"(r512[3]), "=l"(r512[4]), "=l"(r512[5]), "=l"(r512[6]), "=l"(r512[7])
		: "l"(r512[3]), "l"(r512[4]), "l"(r512[5]), "l"(r512[6]), "l"(r512[7]), "l"(t[0]), "l"(t[1]), "l"(t[2]), "l"(t[3]), "l"(t[4])
		: "memory"
	);
	// Reduce from 512 to 320 
	_imm_umul(r512 + 4, 0x1000003D1ULL, t);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %5, %9;\r\n"		
		"addc.cc.u64 %1, %6, %10;\r\n"		
		"addc.cc.u64 %2, %7, %11;\r\n"		
		"addc.cc.u64 %3, %8, %12;\r\n"		
		"addc.u64 %4, 0, 0;\r\n"
		"}\n\t"
		: "=l"(r512[0]), "=l"(r512[1]), "=l"(r512[2]), "=l"(r512[3]), "=l"(c)
		: "l"(r512[0]), "l"(r512[1]), "l"(r512[2]), "l"(r512[3]), "l"(t[0]), "l"(t[1]), "l"(t[2]), "l"(t[3])
		: "memory"
	);
	// Reduce from 320 to 256 
	// No overflow possible here t[4]+c<=0x1000003D1ULL
	asm(
		"{\n\t"
		"mul.lo.u64 %0, %2, %3;\r\n"
		"mul.hi.u64 %1, %2, %3;\r\n"
		"}\n\t"
		: "=l"(al), "=l"(ah)
		: "l"(t[4] + c), "l"(0x1000003D1ULL)
		: "memory"
	);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %4, %8;\r\n"
		"addc.cc.u64 %1, %5, %9;\r\n"
		"addc.cc.u64 %2, %6, 0;\r\n"
		"addc.u64 %3, %7, 0;\r\n"
		"}\n\t"
		: "=l"((*RES).bitsu64[0]), "=l"((*RES).bitsu64[1]), "=l"((*RES).bitsu64[2]), "=l"((*RES).bitsu64[3])
		: "l"(r512[0]), "l"(r512[1]), "l"(r512[2]), "l"(r512[3]), "l"(al), "l"(ah)
		: "memory"
	);
}

__device__ static void __forceinline__ _ModSquareK1(D_Int *RES, D_Int *a) {
	uint64_t r512[8];
	uint64_t u10, u11, c;
	uint64_t t1;
	uint64_t t2;
	uint64_t t[5];

	asm(
		"{\n\t"
		"mul.lo.u64 %0, %2, %3;\r\n"
		"mul.hi.u64 %1, %2, %3;\r\n"
		"}\n\t"
		: "=l"(r512[0]), "=l"(t[1])
		: "l"(a->bitsu64[0]), "l"(a->bitsu64[0])
		: "memory"
	);
	asm(
		"{\n\t"
		"mul.lo.u64 %0, %2, %3;\r\n"
		"mul.hi.u64 %1, %2, %3;\r\n"
		"}\n\t"
		: "=l"(t[3]), "=l"(t[4])
		: "l"(a->bitsu64[0]), "l"(a->bitsu64[1])
		: "memory"
	);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %3, %5;\r\n"
		"addc.cc.u64 %1, %4, %6;\r\n"
		"addc.u64 %2, 0, 0;\r\n"
		"}\n\t"
		: "=l"(t[3]), "=l"(t[4]), "=l"(t1)
		: "l"(t[3]), "l"(t[4]), "l"(t[3]), "l"(t[4])
		: "memory"
	);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %3, %5;\r\n"
		"addc.cc.u64 %1, %4, 0;\r\n"
		"addc.u64 %2, %6, 0;\r\n"
		"}\n\t"
		: "=l"(t[3]), "=l"(t[4]), "=l"(t1)
		: "l"(t[1]), "l"(t[4]), "l"(t[3]), "l"(t1)
		: "memory"
	);
	r512[1] = t[3];
	asm(
		"{\n\t"
		"mul.lo.u64 %0, %2, %3;\r\n"
		"mul.hi.u64 %1, %2, %3;\r\n"
		"}\n\t"
		: "=l"(t[0]), "=l"(t[1])
		: "l"(a->bitsu64[0]), "l"(a->bitsu64[2])
		: "memory"
	);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %3, %5;\r\n"
		"addc.cc.u64 %1, %4, %6;\r\n"
		"addc.u64 %2, 0, 0;\r\n"
		"}\n\t"
		: "=l"(t[0]), "=l"(t[1]), "=l"(t2)
		: "l"(t[0]), "l"(t[1]), "l"(t[0]), "l"(t[1])
		: "memory"
	);
	asm(
		"{\n\t"
		"mul.lo.u64 %0, %2, %3;\r\n"
		"mul.hi.u64 %1, %2, %3;\r\n"
		"}\n\t"
		: "=l"(u10), "=l"(u11)
		: "l"(a->bitsu64[1]), "l"(a->bitsu64[1])
		: "memory"
	);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %3, %5;\r\n"
		"addc.cc.u64 %1, %4, %6;\r\n"
		"addc.u64 %2, %7, 0;\r\n"
		"}\n\t"
		: "=l"(t[0]), "=l"(t[1]), "=l"(t2)
		: "l"(t[0]), "l"(t[1]), "l"(u10), "l"(u11), "l"(t2)
		: "memory"
	);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %3, %5;\r\n"
		"addc.cc.u64 %1, %4, %6;\r\n"
		"addc.u64 %2, %7, 0;\r\n"
		"}\n\t"
		: "=l"(t[0]), "=l"(t[1]), "=l"(t2)
		: "l"(t[0]), "l"(t[1]), "l"(t[4]), "l"(t1), "l"(t2)
		: "memory"
	);
	r512[2] = t[0];
	asm(
		"{\n\t"
		"mul.lo.u64 %0, %2, %3;\r\n"
		"mul.hi.u64 %1, %2, %3;\r\n"
		"}\n\t"
		: "=l"(t[3]), "=l"(t[4])
		: "l"(a->bitsu64[0]), "l"(a->bitsu64[3])
		: "memory"
	);
	asm(
		"{\n\t"
		"mul.lo.u64 %0, %2, %3;\r\n"
		"mul.hi.u64 %1, %2, %3;\r\n"
		"}\n\t"
		: "=l"(u10), "=l"(u11)
		: "l"(a->bitsu64[1]), "l"(a->bitsu64[2])
		: "memory"
	);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %3, %5;\r\n"
		"addc.cc.u64 %1, %4, %6;\r\n"
		"addc.u64 %2, 0, 0;\r\n"
		"}\n\t"
		: "=l"(t[3]), "=l"(t[4]), "=l"(t1)
		: "l"(t[3]), "l"(t[4]), "l"(u10), "l"(u11)
		: "memory"
	);
	t1 += t1;
	asm(
		"{\n\t"
		"add.cc.u64 %0, %3, %5;\r\n"
		"addc.cc.u64 %1, %4, %6;\r\n"
		"addc.u64 %2, %7, 0;\r\n"
		"}\n\t"
		: "=l"(t[3]), "=l"(t[4]), "=l"(t1)
		: "l"(t[3]), "l"(t[4]), "l"(t[3]), "l"(t[4]), "l"(t1)
		: "memory"
	);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %3, %5;\r\n"
		"addc.cc.u64 %1, %4, %6;\r\n"
		"addc.u64 %2, %7, 0;\r\n"
		"}\n\t"
		: "=l"(t[3]), "=l"(t[4]), "=l"(t1)
		: "l"(t[3]), "l"(t[4]), "l"(t[1]), "l"(t2), "l"(t1)
		: "memory"
	);
	r512[3] = t[3];
	asm(
		"{\n\t"
		"mul.lo.u64 %0, %2, %3;\r\n"
		"mul.hi.u64 %1, %2, %3;\r\n"
		"}\n\t"
		: "=l"(t[0]), "=l"(t[1])
		: "l"(a->bitsu64[1]), "l"(a->bitsu64[3])
		: "memory"
	);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %3, %5;\r\n"
		"addc.cc.u64 %1, %4, %6;\r\n"
		"addc.u64 %2, 0, 0;\r\n"
		"}\n\t"
		: "=l"(t[0]), "=l"(t[1]), "=l"(t2)
		: "l"(t[0]), "l"(t[1]), "l"(t[0]), "l"(t[1])
		: "memory"
	);
	asm(
		"{\n\t"
		"mul.lo.u64 %0, %2, %3;\r\n"
		"mul.hi.u64 %1, %2, %3;\r\n"
		"}\n\t"
		: "=l"(u10), "=l"(u11)
		: "l"(a->bitsu64[2]), "l"(a->bitsu64[2])
		: "memory"
	);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %3, %5;\r\n"
		"addc.cc.u64 %1, %4, %6;\r\n"
		"addc.u64 %2, %7, 0;\r\n"
		"}\n\t"
		: "=l"(t[0]), "=l"(t[1]), "=l"(t2)
		: "l"(t[0]), "l"(t[1]), "l"(u10), "l"(u11), "l"(t2)
		: "memory"
	);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %3, %5;\r\n"
		"addc.cc.u64 %1, %4, %6;\r\n"
		"addc.u64 %2, %7, 0;\r\n"
		"}\n\t"
		: "=l"(t[0]), "=l"(t[1]), "=l"(t2)
		: "l"(t[0]), "l"(t[1]), "l"(t[4]), "l"(t1), "l"(t2)
		: "memory"
	);
	r512[4] = t[0];
	asm(
		"{\n\t"
		"mul.lo.u64 %0, %2, %3;\r\n"
		"mul.hi.u64 %1, %2, %3;\r\n"
		"}\n\t"
		: "=l"(t[3]), "=l"(t[4])
		: "l"(a->bitsu64[2]), "l"(a->bitsu64[3])
		: "memory"
	);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %3, %5;\r\n"
		"addc.cc.u64 %1, %4, %6;\r\n"
		"addc.u64 %2, 0, 0;\r\n"
		"}\n\t"
		: "=l"(t[3]), "=l"(t[4]), "=l"(t1)
		: "l"(t[3]), "l"(t[4]), "l"(t[3]), "l"(t[4])
		: "memory"
	);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %3, %5;\r\n"
		"addc.cc.u64 %1, %4, %6;\r\n"
		"addc.u64 %2, %7, 0;\r\n"
		"}\n\t"
		: "=l"(t[3]), "=l"(t[4]), "=l"(t1)
		: "l"(t[3]), "l"(t[4]), "l"(t[1]), "l"(t2), "l"(t1)
		: "memory"
	);
	r512[5] = t[3];
	asm(
		"{\n\t"
		"mul.lo.u64 %0, %2, %3;\r\n"
		"mul.hi.u64 %1, %2, %3;\r\n"
		"}\n\t"
		: "=l"(t[0]), "=l"(t[1])
		: "l"(a->bitsu64[3]), "l"(a->bitsu64[3])
		: "memory"
	);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %2, %4;\r\n"
		"addc.u64 %1, %3, %5;\r\n"
		"}\n\t"
		: "=l"(t[0]), "=l"(t[1])
		: "l"(t[0]), "l"(t[1]), "l"(t[4]), "l"(t1)
		: "memory"
	);
	r512[6] = t[0];
	r512[7] = t[1];
	// Reduce from 512 to 320 
	// Reduce from 512 to 320 
	_imm_umul(r512 + 4, 0x1000003D1ULL, t);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %5, %9;\r\n"		
		"addc.cc.u64 %1, %6, %10;\r\n"		
		"addc.cc.u64 %2, %7, %11;\r\n"		
		"addc.cc.u64 %3, %8, %12;\r\n"		
		"addc.u64 %4, 0, 0;\r\n"
		"}\n\t"
		: "=l"(r512[0]), "=l"(r512[1]), "=l"(r512[2]), "=l"(r512[3]), "=l"(c)
		: "l"(r512[0]), "l"(r512[1]), "l"(r512[2]), "l"(r512[3]), "l"(t[0]), "l"(t[1]), "l"(t[2]), "l"(t[3])
		: "memory"
	);
	// Reduce from 320 to 256 
	// No overflow possible here t[4]+c<=0x1000003D1ULL
	asm(
		"{\n\t"
		"mul.lo.u64 %0, %2, %3;\r\n"
		"mul.hi.u64 %1, %2, %3;\r\n"
		"}\n\t"
		: "=l"(u10), "=l"(u11)
		: "l"(t[4] + c), "l"(0x1000003D1ULL)
		: "memory"
	);
	asm(
		"{\n\t"
		"add.cc.u64 %0, %4, %8;\r\n"
		"addc.cc.u64 %1, %5, %9;\r\n"
		"addc.cc.u64 %2, %6, 0;\r\n"
		"addc.u64 %3, %7, 0;\r\n"
		"}\n\t"
		: "=l"((*RES).bitsu64[0]), "=l"((*RES).bitsu64[1]), "=l"((*RES).bitsu64[2]), "=l"((*RES).bitsu64[3])
		: "l"(r512[0]), "l"(r512[1]), "l"(r512[2]), "l"(r512[3]), "l"(u10), "l"(u11)
		: "memory"
	);
	// Probability of carry here or that this>P is very very unlikely
	RES->bitsu64[4] = 0ULL;
}

__device__ static void __forceinline__ _ModSub(D_Int *RES, D_Int *a, D_Int *b) {
	_Sub(RES, a, b);
	if (RES->bits64[NB64BLOCK - 1] < 0)
		_Add(RES, &_P);
}

__device__ static void __forceinline__ _ModSub(D_Int *RES, D_Int *a) {
	_Sub(RES, a);
	if (RES->bits64[NB64BLOCK - 1] < 0)
		_Add(RES, &_P);
}

__device__ static void __forceinline__ _ModAdd(D_Int *RES, D_Int *a, D_Int *b) {
	D_Int p;
	
	_Add(RES, a, b);
	_Sub(&p, RES, &_P);
	if(p.bits64[NB64BLOCK - 1] >= 0)
		_SetD_Int(RES, &p);
}

__device__ static void __forceinline__ _ModInv(D_Int *RES) {
	D_Int u;
	D_Int v;
	D_Int r;
	D_Int s;
	D_Int r0_P;
	D_Int s0_P;
	D_Int uu_u;
	D_Int uv_v;
	D_Int vu_u;
	D_Int vv_v;
	D_Int uu_r;
	D_Int uv_s;
	D_Int vu_r;
	D_Int vv_s;
	D_Int checkGE;
	int64_t bitCount;
	int64_t uu, uv, vu, vv;
	int64_t v0, u0;
	int64_t nb0;

	_SetD_Int(&u,&_P);
	_SetD_Int(&v,RES);
	_ClearD_Int(&r);
	_SetD_Int_One(&s);
	while (!IS_ZERO(u)) {
		uu = 1; uv = 0;
		vu = 0; vv = 1;
		u0 = u.bits64[0];
		v0 = v.bits64[0];
		bitCount = 0;
		while (true) {
			while (IS_EVEN(u0) && bitCount<62) {
				bitCount++;
				u0 >>= 1;
				vu <<= 1;
				vv <<= 1;
			}
			if (bitCount == 62)
				break;
			nb0 = (v0 + u0) & 0x3;
			if (nb0 == 0) {
				SWAP_ADD(uv, vv);
				SWAP_ADD(uu, vu);
				SWAP_ADD(u0, v0);
			} else {
				SWAP_SUB(uv, vv);
				SWAP_SUB(uu, vu);
				SWAP_SUB(u0, v0);
			}
		}
		_IMult(&uu_u,&u,uu);
		_IMult(&uv_v,&v,uv);
		_IMult(&vu_u,&u,vu);
		_IMult(&vv_v,&v,vv);
		_IMult(&uu_r,&r,uu);
		_IMult(&uv_s,&s,uv);
		_IMult(&vu_r,&r,vu);
		_IMult(&vv_s,&s,vv);
		uint64_t r0 = ((uu_r.bitsu64[0] + uv_s.bitsu64[0]) * MM64) & MSK62;
		uint64_t s0 = ((vu_r.bitsu64[0] + vv_s.bitsu64[0]) * MM64) & MSK62;
		_Mult(&r0_P,&_P,r0);
		_Mult(&s0_P,&_P,s0);
		_Add(&u,&uu_u,&uv_v);
		_Add(&v,&vu_u,&vv_v);
		_Add(&r,&uu_r,&uv_s);
		_Add(&r,&r0_P);
		_Add(&s,&vu_r,&vv_s);
		_Add(&s,&s0_P);
		_shiftR(62, &u);
		_shiftR(62, &v);
		_shiftR(62, &r);
		_shiftR(62, &s);
	}
	if (IS_NEGATIVE(v)) {
		_Neg(&v);
		_Neg(&s);
		_Add(&s,&_P);
	}
	if (!IS_ONE(v)) {
		_ClearD_Int(RES);
		return;
	}
	if (IS_NEGATIVE(s))
		_Add(&s,&_P);
	_Sub(&checkGE, &s, &_P);
	if (IS_POSITIVE(checkGE))
		_Sub(&s,&_P);
	_SetD_Int(RES, &s);
}

__device__ static void __forceinline__ _PointAdd2(D_Point *RES, D_Point *p1, D_Point *p2) {
	D_Int u;
	D_Int v;
	D_Int u1;
	D_Int v1;
	D_Int vs2;
	D_Int vs3;
	D_Int us2;
	D_Int a;
	D_Int us2w;
	D_Int vs2v2;
	D_Int vs3u2;
	D_Int _2vs2v2;
	
	_ModMulK1(&u1,&((*p2).Y), &((*p1).Z));
	_ModMulK1(&v1, &((*p2).X), &((*p1).Z));
	_ModSub(&u,&u1, &((*p1).Y));
	_ModSub(&v,&v1, &((*p1).X));
	_ModSquareK1(&us2,&u);
	_ModSquareK1(&vs2,&v);
	_ModMulK1(&vs3,&vs2, &v);
	_ModMulK1(&us2w,&us2, &((*p1).Z));
	_ModMulK1(&vs2v2,&vs2, &((*p1).X));
	_ModAdd(&_2vs2v2,&vs2v2, &vs2v2);
	_ModSub(&a,&us2w, &vs3);
	_ModSub(&a,&_2vs2v2);
	_ModMulK1(&((*RES).X),&v, &a);
	_ModMulK1(&vs3u2,&vs3, &((*p1).Y));
	_ModSub(&((*RES).Y),&vs2v2, &a);
	_ModMulK1(&((*RES).Y), &u);
	_ModSub(&((*RES).Y),&vs3u2);
	_ModMulK1(&((*RES).Z),&vs3, &((*p1).Z));
}

__device__ static void __noinline__ _ComputePublicKey(
	D_Int *Q, 
	D_Point *R, 
	D_Point *publicKey, 
	D_Int256 *subp,
	D_Int *inverse,
	uint32_t COUNT 
) {
	uint8_t b;
	D_Int PP;
	uint8_t i = 0x00;
	uint8_t found = 0x00;
	
	_ClearD_Point(publicKey);
#pragma unroll
	for (i = 0; i < RANDOMBYTE; i++) {
		b = (*Q).uc[i];
		if (b) {
			found = 0x01;
			break;
		}
	}
	if (found) {
		_SetD_Point(publicKey, &(GTable[256 * i + (b-1)]));
		i++;
		for(; i < RANDOMBYTE; i++) {
			b = (*Q).uc[i];
			if (b) {
				_PointAdd2(publicKey, publicKey, &GTable[256 * i + (b-1)]);
			}
		}
		_PointAdd2(publicKey, publicKey, R);
		_ModInv(&((*publicKey).Z));
		_ModMulK1(&((*publicKey).X),&((*publicKey).Z));
		_ModMulK1(&((*publicKey).Y),&((*publicKey).Z));
		_SetD_Int_One(&((*publicKey).Z));
	} else {
		_SetD_Point(publicKey, R);
	}
#pragma unroll
	for (uint32_t zzloop=1;zzloop<COUNT;zzloop++) {
		_ModSub(&PP, &((GTableAdds[zzloop-1]).X), &((*publicKey).X));
		if (zzloop == 1) {
			_SetD_Int(&(subp[zzloop-1]), &PP);
		} else {
			if (zzloop == (COUNT-1)) {
				_ModMulK1(inverse, &(subp[zzloop-2]), &PP);
			} else {
				_ModMulK1(&(subp[zzloop-1]), &(subp[zzloop-2]), &PP);
			}
		}
	}
}

#endif
