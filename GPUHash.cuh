#ifndef GPUHASH_H
#define GPUHASH_H

#include <cuda_runtime.h>
#include <cstdint>

// ---------------------------------------------------------------------------------
// SHA256 CONSTANTS
// ---------------------------------------------------------------------------------

__device__ __constant__ uint32_t K[64] = {
	0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
	0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
	0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
	0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
	0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
	0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
	0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
	0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2
};

// ---------------------------------------------------------------------------------
// ROTATE & SIGMA FUNCTIONS (OPTIMIZED WITH PTX FUNNEL SHIFT & LOP3)
// ---------------------------------------------------------------------------------

__device__ __forceinline__ uint32_t ROR32(uint32_t x, uint32_t n) {
    return __funnelshift_r(x, x, n);
}

__device__ __forceinline__ uint32_t ROL32(uint32_t x, uint32_t n) {
    return __funnelshift_l(x, x, n);
}

__device__ __forceinline__ uint32_t S0(uint32_t x) {
    return ROR32(x, 2) ^ ROR32(x, 13) ^ ROR32(x, 22);
}

__device__ __forceinline__ uint32_t S1(uint32_t x) {
    return ROR32(x, 6) ^ ROR32(x, 11) ^ ROR32(x, 25);
}

__device__ __forceinline__ uint32_t s0(uint32_t x) {
    return ROR32(x, 7) ^ ROR32(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ uint32_t s1(uint32_t x) {
    return ROR32(x, 17) ^ ROR32(x, 19) ^ (x >> 10);
}

// Ch(x, y, z) = z ^ (x & (y ^ z)) --> LUT 0xCA
__device__ __forceinline__ uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) {
    uint32_t ret;
    asm("lop3.b32 %0, %1, %2, %3, 0xCA;" : "=r"(ret) : "r"(x), "r"(y), "r"(z));
    return ret;
}

// Maj(x, y, z) = (x & y) | (z & (x | y)) --> LUT 0xE8
__device__ __forceinline__ uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) {
    uint32_t ret;
    asm("lop3.b32 %0, %1, %2, %3, 0xE8;" : "=r"(ret) : "r"(x), "r"(y), "r"(z));
    return ret;
}

#define S2Round(a, b, c, d, e, f, g, h, k, w) \
    t1 = h + S1(e) + Ch(e, f, g) + k + (w); \
    t2 = S0(a) + Maj(a, b, c); \
    d += t1; \
    h = t1 + t2;

#define WMIX() { \
    w[0]  += s1(w[14]) + w[9]  + s0(w[1]);  \
    w[1]  += s1(w[15]) + w[10] + s0(w[2]);  \
    w[2]  += s1(w[0])  + w[11] + s0(w[3]);  \
    w[3]  += s1(w[1])  + w[12] + s0(w[4]);  \
    w[4]  += s1(w[2])  + w[13] + s0(w[5]);  \
    w[5]  += s1(w[3])  + w[14] + s0(w[6]);  \
    w[6]  += s1(w[4])  + w[15] + s0(w[7]);  \
    w[7]  += s1(w[5])  + w[0]  + s0(w[8]);  \
    w[8]  += s1(w[6])  + w[1]  + s0(w[9]);  \
    w[9]  += s1(w[7])  + w[2]  + s0(w[10]); \
    w[10] += s1(w[8])  + w[3]  + s0(w[11]); \
    w[11] += s1(w[9])  + w[4]  + s0(w[12]); \
    w[12] += s1(w[10]) + w[5]  + s0(w[13]); \
    w[13] += s1(w[11]) + w[6]  + s0(w[14]); \
    w[14] += s1(w[12]) + w[7]  + s0(w[15]); \
    w[15] += s1(w[13]) + w[8]  + s0(w[0]);  \
}

#define SHA256_RND(k) { \
    S2Round(a, b, c, d, e, f, g, h, K[k],     w[0]);  \
    S2Round(h, a, b, c, d, e, f, g, K[k + 1], w[1]);  \
    S2Round(g, h, a, b, c, d, e, f, K[k + 2], w[2]);  \
    S2Round(f, g, h, a, b, c, d, e, K[k + 3], w[3]);  \
    S2Round(e, f, g, h, a, b, c, d, K[k + 4], w[4]);  \
    S2Round(d, e, f, g, h, a, b, c, K[k + 5], w[5]);  \
    S2Round(c, d, e, f, g, h, a, b, K[k + 6], w[6]);  \
    S2Round(b, c, d, e, f, g, h, a, K[k + 7], w[7]);  \
    S2Round(a, b, c, d, e, f, g, h, K[k + 8], w[8]);  \
    S2Round(h, a, b, c, d, e, f, g, K[k + 9], w[9]);  \
    S2Round(g, h, a, b, c, d, e, f, K[k + 10], w[10]);\
    S2Round(f, g, h, a, b, c, d, e, K[k + 11], w[11]);\
    S2Round(e, f, g, h, a, b, c, d, K[k + 12], w[12]);\
    S2Round(d, e, f, g, h, a, b, c, K[k + 13], w[13]);\
    S2Round(c, d, e, f, g, h, a, b, K[k + 14], w[14]);\
    S2Round(b, c, d, e, f, g, h, a, K[k + 15], w[15]);\
}

#define bswap32(v) __byte_perm(v, 0, 0x0123)

__device__ __forceinline__ void SHA256Transform(uint32_t s[8], uint32_t* __restrict__ w) {
	uint32_t t1, t2;

	uint32_t a = s[0], b = s[1], c = s[2], d = s[3];
	uint32_t e = s[4], f = s[5], g = s[6], h = s[7];

	SHA256_RND(0);
	WMIX();
	SHA256_RND(16);
	WMIX();
	SHA256_RND(32);
	WMIX();
	SHA256_RND(48);

	s[0] += a; s[1] += b; s[2] += c; s[3] += d;
	s[4] += e; s[5] += f; s[6] += g; s[7] += h;
}

// ---------------------------------------------------------------------------------
// RIPEMD160
// ---------------------------------------------------------------------------------

#define f1(x, y, z) ((x) ^ (y) ^ (z))
#define f2(x, y, z) (((x) & (y)) | (~(x) & (z)))
#define f3(x, y, z) (((x) | ~(y)) ^ (z))
#define f4(x, y, z) (((x) & (z)) | (~(z) & (y)))
#define f5(x, y, z) ((x) ^ ((y) | ~(z)))

#define RPRound(a, b, c, d, e, f, x, k, r) \
    u = a + f + x + k; \
    a = ROL32(u, r) + e; \
    c = ROL32(c, 10);

#define R11(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f1(b, c, d), x, 0, r)
#define R21(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f2(b, c, d), x, 0x5A827999ul, r)
#define R31(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f3(b, c, d), x, 0x6ED9EBA1ul, r)
#define R41(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f4(b, c, d), x, 0x8F1BBCDCul, r)
#define R51(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f5(b, c, d), x, 0xA953FD4Eul, r)
#define R12(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f5(b, c, d), x, 0x50A28BE6ul, r)
#define R22(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f4(b, c, d), x, 0x5C4DD124ul, r)
#define R32(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f3(b, c, d), x, 0x6D703EF3ul, r)
#define R42(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f2(b, c, d), x, 0x7A6D76E9ul, r)
#define R52(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f1(b, c, d), x, 0, r)

__device__ __forceinline__ void RIPEMD160Transform(uint32_t s[5], const uint32_t* __restrict__ w) {
	uint32_t u;
	uint32_t a1 = s[0], b1 = s[1], c1 = s[2], d1 = s[3], e1 = s[4];
	uint32_t a2 = a1, b2 = b1, c2 = c1, d2 = d1, e2 = e1;

	R11(a1, b1, c1, d1, e1, w[0], 11);
	R12(a2, b2, c2, d2, e2, w[5], 8);
	R11(e1, a1, b1, c1, d1, w[1], 14);
	R12(e2, a2, b2, c2, d2, w[14], 9);
	R11(d1, e1, a1, b1, c1, w[2], 15);
	R12(d2, e2, a2, b2, c2, w[7], 9);
	R11(c1, d1, e1, a1, b1, w[3], 12);
	R12(c2, d2, e2, a2, b2, w[0], 11);
	R11(b1, c1, d1, e1, a1, w[4], 5);
	R12(b2, c2, d2, e2, a2, w[9], 13);
	R11(a1, b1, c1, d1, e1, w[5], 8);
	R12(a2, b2, c2, d2, e2, w[2], 15);
	R11(e1, a1, b1, c1, d1, w[6], 7);
	R12(e2, a2, b2, c2, d2, w[11], 15);
	R11(d1, e1, a1, b1, c1, w[7], 9);
	R12(d2, e2, a2, b2, c2, w[4], 5);
	R11(c1, d1, e1, a1, b1, w[8], 11);
	R12(c2, d2, e2, a2, b2, w[13], 7);
	R11(b1, c1, d1, e1, a1, w[9], 13);
	R12(b2, c2, d2, e2, a2, w[6], 7);
	R11(a1, b1, c1, d1, e1, w[10], 14);
	R12(a2, b2, c2, d2, e2, w[15], 8);
	R11(e1, a1, b1, c1, d1, w[11], 15);
	R12(e2, a2, b2, c2, d2, w[8], 11);
	R11(d1, e1, a1, b1, c1, w[12], 6);
	R12(d2, e2, a2, b2, c2, w[1], 14);
	R11(c1, d1, e1, a1, b1, w[13], 7);
	R12(c2, d2, e2, a2, b2, w[10], 14);
	R11(b1, c1, d1, e1, a1, w[14], 9);
	R12(b2, c2, d2, e2, a2, w[3], 12);
	R11(a1, b1, c1, d1, e1, w[15], 8);
	R12(a2, b2, c2, d2, e2, w[12], 6);

	R21(e1, a1, b1, c1, d1, w[7], 7);
	R22(e2, a2, b2, c2, d2, w[6], 9);
	R21(d1, e1, a1, b1, c1, w[4], 6);
	R22(d2, e2, a2, b2, c2, w[11], 13);
	R21(c1, d1, e1, a1, b1, w[13], 8);
	R22(c2, d2, e2, a2, b2, w[3], 15);
	R21(b1, c1, d1, e1, a1, w[1], 13);
	R22(b2, c2, d2, e2, a2, w[7], 7);
	R21(a1, b1, c1, d1, e1, w[10], 11);
	R22(a2, b2, c2, d2, e2, w[0], 12);
	R21(e1, a1, b1, c1, d1, w[6], 9);
	R22(e2, a2, b2, c2, d2, w[13], 8);
	R21(d1, e1, a1, b1, c1, w[15], 7);
	R22(d2, e2, a2, b2, c2, w[5], 9);
	R21(c1, d1, e1, a1, b1, w[3], 15);
	R22(c2, d2, e2, a2, b2, w[10], 11);
	R21(b1, c1, d1, e1, a1, w[12], 7);
	R22(b2, c2, d2, e2, a2, w[14], 7);
	R21(a1, b1, c1, d1, e1, w[0], 12);
	R22(a2, b2, c2, d2, e2, w[15], 7);
	R21(e1, a1, b1, c1, d1, w[9], 15);
	R22(e2, a2, b2, c2, d2, w[8], 12);
	R21(d1, e1, a1, b1, c1, w[5], 9);
	R22(d2, e2, a2, b2, c2, w[12], 7);
	R21(c1, d1, e1, a1, b1, w[2], 11);
	R22(c2, d2, e2, a2, b2, w[4], 6);
	R21(b1, c1, d1, e1, a1, w[14], 7);
	R22(b2, c2, d2, e2, a2, w[9], 15);
	R21(a1, b1, c1, d1, e1, w[11], 13);
	R22(a2, b2, c2, d2, e2, w[1], 13);
	R21(e1, a1, b1, c1, d1, w[8], 12);
	R22(e2, a2, b2, c2, d2, w[2], 11);

	R31(d1, e1, a1, b1, c1, w[3], 11);
	R32(d2, e2, a2, b2, c2, w[15], 9);
	R31(c1, d1, e1, a1, b1, w[10], 13);
	R32(c2, d2, e2, a2, b2, w[5], 7);
	R31(b1, c1, d1, e1, a1, w[14], 6);
	R32(b2, c2, d2, e2, a2, w[1], 15);
	R31(a1, b1, c1, d1, e1, w[4], 7);
	R32(a2, b2, c2, d2, e2, w[3], 11);
	R31(e1, a1, b1, c1, d1, w[9], 14);
	R32(e2, a2, b2, c2, d2, w[7], 8);
	R31(d1, e1, a1, b1, c1, w[15], 9);
	R32(d2, e2, a2, b2, c2, w[14], 6);
	R31(c1, d1, e1, a1, b1, w[8], 13);
	R32(c2, d2, e2, a2, b2, w[6], 6);
	R31(b1, c1, d1, e1, a1, w[1], 15);
	R32(b2, c2, d2, e2, a2, w[9], 14);
	R31(a1, b1, c1, d1, e1, w[2], 14);
	R32(a2, b2, c2, d2, e2, w[11], 12);
	R31(e1, a1, b1, c1, d1, w[7], 8);
	R32(e2, a2, b2, c2, d2, w[8], 13);
	R31(d1, e1, a1, b1, c1, w[0], 13);
	R32(d2, e2, a2, b2, c2, w[12], 5);
	R31(c1, d1, e1, a1, b1, w[6], 6);
	R32(c2, d2, e2, a2, b2, w[2], 14);
	R31(b1, c1, d1, e1, a1, w[13], 5);
	R32(b2, c2, d2, e2, a2, w[10], 13);
	R31(a1, b1, c1, d1, e1, w[11], 12);
	R32(a2, b2, c2, d2, e2, w[0], 13);
	R31(e1, a1, b1, c1, d1, w[5], 7);
	R32(e2, a2, b2, c2, d2, w[4], 7);
	R31(d1, e1, a1, b1, c1, w[12], 5);
	R32(d2, e2, a2, b2, c2, w[13], 5);

	R41(c1, d1, e1, a1, b1, w[1], 11);
	R42(c2, d2, e2, a2, b2, w[8], 15);
	R41(b1, c1, d1, e1, a1, w[9], 12);
	R42(b2, c2, d2, e2, a2, w[6], 5);
	R41(a1, b1, c1, d1, e1, w[11], 14);
	R42(a2, b2, c2, d2, e2, w[4], 8);
	R41(e1, a1, b1, c1, d1, w[10], 15);
	R42(e2, a2, b2, c2, d2, w[1], 11);
	R41(d1, e1, a1, b1, c1, w[0], 14);
	R42(d2, e2, a2, b2, c2, w[3], 14);
	R41(c1, d1, e1, a1, b1, w[8], 15);
	R42(c2, d2, e2, a2, b2, w[11], 14);
	R41(b1, c1, d1, e1, a1, w[12], 9);
	R42(b2, c2, d2, e2, a2, w[15], 6);
	R41(a1, b1, c1, d1, e1, w[4], 8);
	R42(a2, b2, c2, d2, e2, w[0], 14);
	R41(e1, a1, b1, c1, d1, w[13], 9);
	R42(e2, a2, b2, c2, d2, w[5], 6);
	R41(d1, e1, a1, b1, c1, w[3], 14);
	R42(d2, e2, a2, b2, c2, w[12], 9);
	R41(c1, d1, e1, a1, b1, w[7], 5);
	R42(c2, d2, e2, a2, b2, w[2], 12);
	R41(b1, c1, d1, e1, a1, w[15], 6);
	R42(b2, c2, d2, e2, a2, w[13], 9);
	R41(a1, b1, c1, d1, e1, w[14], 8);
	R42(a2, b2, c2, d2, e2, w[9], 12);
	R41(e1, a1, b1, c1, d1, w[5], 6);
	R42(e2, a2, b2, c2, d2, w[7], 5);
	R41(d1, e1, a1, b1, c1, w[6], 5);
	R42(d2, e2, a2, b2, c2, w[10], 15);
	R41(c1, d1, e1, a1, b1, w[2], 12);
	R42(c2, d2, e2, a2, b2, w[14], 8);

	R51(b1, c1, d1, e1, a1, w[4], 9);
	R52(b2, c2, d2, e2, a2, w[12], 8);
	R51(a1, b1, c1, d1, e1, w[0], 15);
	R52(a2, b2, c2, d2, e2, w[15], 5);
	R51(e1, a1, b1, c1, d1, w[5], 5);
	R52(e2, a2, b2, c2, d2, w[10], 12);
	R51(d1, e1, a1, b1, c1, w[9], 11);
	R52(d2, e2, a2, b2, c2, w[4], 9);
	R51(c1, d1, e1, a1, b1, w[7], 6);
	R52(c2, d2, e2, a2, b2, w[1], 12);
	R51(b1, c1, d1, e1, a1, w[12], 8);
	R52(b2, c2, d2, e2, a2, w[5], 5);
	R51(a1, b1, c1, d1, e1, w[2], 13);
	R52(a2, b2, c2, d2, e2, w[8], 14);
	R51(e1, a1, b1, c1, d1, w[10], 12);
	R52(e2, a2, b2, c2, d2, w[7], 6);
	R51(d1, e1, a1, b1, c1, w[14], 5);
	R52(d2, e2, a2, b2, c2, w[6], 8);
	R51(c1, d1, e1, a1, b1, w[1], 12);
	R52(c2, d2, e2, a2, b2, w[2], 13);
	R51(b1, c1, d1, e1, a1, w[3], 13);
	R52(b2, c2, d2, e2, a2, w[13], 6);
	R51(a1, b1, c1, d1, e1, w[8], 14);
	R52(a2, b2, c2, d2, e2, w[14], 5);
	R51(e1, a1, b1, c1, d1, w[11], 11);
	R52(e2, a2, b2, c2, d2, w[0], 15);
	R51(d1, e1, a1, b1, c1, w[6], 8);
	R52(d2, e2, a2, b2, c2, w[3], 13);
	R51(c1, d1, e1, a1, b1, w[15], 5);
	R52(c2, d2, e2, a2, b2, w[9], 11);
	R51(b1, c1, d1, e1, a1, w[13], 6);
	R52(b2, c2, d2, e2, a2, w[11], 11);

	uint32_t t = s[0];
	s[0] = s[1] + c1 + d2;
	s[1] = s[2] + d1 + e2;
	s[2] = s[3] + e1 + a2;
	s[3] = s[4] + a1 + b2;
	s[4] = t + b1 + c2;
}

// ---------------------------------------------------------------------------------
// Key Encoding / Hash160 (FULLY REGISTER-OPTIMIZED)
// ---------------------------------------------------------------------------------

__device__ __forceinline__ void _GetHash160Comp(const uint64_t* __restrict__ x, uint8_t isOdd, uint8_t* __restrict__ hash) {
	const uint32_t* x32 = (const uint32_t*)(x);
	uint32_t publicKeyBytes[16];
	uint32_t s[16];

	// Direct byte extraction via PTX byte-permute (PRMT)
	publicKeyBytes[0] = __byte_perm(x32[7], 0x2 + isOdd, 0x4321);
	publicKeyBytes[1] = __byte_perm(x32[7], x32[6], 0x0765);
	publicKeyBytes[2] = __byte_perm(x32[6], x32[5], 0x0765);
	publicKeyBytes[3] = __byte_perm(x32[5], x32[4], 0x0765);
	publicKeyBytes[4] = __byte_perm(x32[4], x32[3], 0x0765);
	publicKeyBytes[5] = __byte_perm(x32[3], x32[2], 0x0765);
	publicKeyBytes[6] = __byte_perm(x32[2], x32[1], 0x0765);
	publicKeyBytes[7] = __byte_perm(x32[1], x32[0], 0x0765);
	publicKeyBytes[8] = __byte_perm(x32[0], 0x80, 0x0456);

	// Zero-fill padding & length directly
	publicKeyBytes[9]  = 0;
	publicKeyBytes[10] = 0;
	publicKeyBytes[11] = 0;
	publicKeyBytes[12] = 0;
	publicKeyBytes[13] = 0;
	publicKeyBytes[14] = 0;
	publicKeyBytes[15] = 0x108; // 264 bits (33 bytes)

	// Inlined Initial State SHA-256
	s[0] = 0x6a09e667ul; s[1] = 0xbb67ae85ul; s[2] = 0x3c6ef372ul; s[3] = 0xa54ff53aul;
	s[4] = 0x510e527ful; s[5] = 0x9b05688cul; s[6] = 0x1f83d9abul; s[7] = 0x5be0cd19ul;

	SHA256Transform(s, publicKeyBytes);

	// Unrolled Big-Endian to Little-Endian conversion for RIPEMD
	s[0] = bswap32(s[0]);
	s[1] = bswap32(s[1]);
	s[2] = bswap32(s[2]);
	s[3] = bswap32(s[3]);
	s[4] = bswap32(s[4]);
	s[5] = bswap32(s[5]);
	s[6] = bswap32(s[6]);
	s[7] = bswap32(s[7]);

	// RIPEMD Padding Block
	s[8]  = 0x80;
	s[9]  = 0;
	s[10] = 0;
	s[11] = 0;
	s[12] = 0;
	s[13] = 0;
	s[14] = 256; // 32 bytes (256 bits)
	s[15] = 0;

	// Compute RIPEMD160 in local registers
	uint32_t rState[5];
	rState[0] = 0x67452301ul;
	rState[1] = 0xEFCDAB89ul;
	rState[2] = 0x98BADCFEul;
	rState[3] = 0x10325476ul;
	rState[4] = 0xC3D2E1F0ul;

	RIPEMD160Transform(rState, s);

	// Single coalesced write to output buffer
	uint32_t* hash32 = (uint32_t*)hash;
	hash32[0] = rState[0];
	hash32[1] = rState[1];
	hash32[2] = rState[2];
	hash32[3] = rState[3];
	hash32[4] = rState[4];
}

#endif
