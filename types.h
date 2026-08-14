#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NB64BLOCK				5
#define KEYBYTESCOUNT			9

#define KEYTOFINDCOUNT			4294967296
#define GROUPCOUNT				2048
#define SIZET					KEYTOFINDCOUNT/GROUPCOUNT
#define	RANDOMBYTE				4
#define MINUSPRINTING			0

#define TEST
#undef  TEST

#ifdef TEST
	#define MINPREFIX			0x10
	#define MAXPREFIX			0x1f
	const char rmdHex[] = 		"61eb8a50c86b0584bb727dd65bed8d2400d6d5aa";
#else
	#define MINPREFIX			0x40
	#define MAXPREFIX			0x7f
	const char rmdHex[] = 		"f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8";
#endif

#define MSK62					0x3FFFFFFFFFFFFFFF

typedef union {
	uint64_t 					bitsu64[NB64BLOCK];
	int64_t  					bits64[NB64BLOCK];
	uint32_t 					bitsu32[NB64BLOCK*2];
	unsigned char 				uc[NB64BLOCK*2*4];
} D_Int;

typedef union {
	uint64_t					bitsu64[(NB64BLOCK-1)];
	int64_t						bits64[(NB64BLOCK-1)];
	uint32_t					bitsu32[(NB64BLOCK-1)*2];
	unsigned char				uc[(NB64BLOCK-1)*2*4];
} D_Int256;

typedef struct {
	D_Int X;
	D_Int Y;
	D_Int Z;
} D_Point;

typedef unsigned char			D_HashRmd[20];
typedef unsigned char			D_CompressedPublicKey[33];

typedef struct {
	D_Int						privateKey;
	D_Point						publicKey;
	D_Int256					subp[GROUPCOUNT-2];
	D_Int						inverse;
	D_HashRmd					rmd;
	uint32_t					indexKeyFound;
} D_KeyPair;

typedef struct {
	uint8_t						keyFound;
	D_Int						privateKey;
	D_HashRmd					rmd160;
} D_Result;

#ifdef __cplusplus
}
#endif

#endif
