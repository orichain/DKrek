#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <immintrin.h>
#include <sys/random.h>
#include <errno.h>

#include <cuda_runtime.h>
#include "kernel.h"
#include "CPUSECP256K1.h"

time_t start_time;

char *format_time(double seconds) {
    static char buffer[50];
    if (seconds < 60)
        snprintf(buffer, sizeof(buffer), "%.1f s", seconds);
    else if (seconds < 3600)
        snprintf(buffer, sizeof(buffer), "%dm %ds", (int)seconds / 60, (int)seconds % 60);
    else
        snprintf(buffer, sizeof(buffer), "%dh %dm", (int)seconds / 3600, ((int)seconds % 3600) / 60);
    return buffer;
}

int generate_random(uint8_t *buffer, size_t length) {
	if (!buffer || length == 0) return -1;
	size_t fetched = 0;
	while (fetched < length) {
		ssize_t result = getrandom(buffer + fetched, length - fetched, 0);
		if (result < 0) {
			if (errno == EINTR) continue;
			return -1;
		}
		fetched += result;
	}
    return 0;
}

int hexchr2bin(const char hex, char *out) {
    if (!out) return 0;
    if (hex >= '0' && hex <= '9') *out = hex - '0';
    else if (hex >= 'A' && hex <= 'F') *out = hex - 'A' + 10;
    else if (hex >= 'a' && hex <= 'f') *out = hex - 'a' + 10;
    else return 0;
    return 1;
}

int hexs2bin(const char *hex, unsigned char *out) {
    if (!hex || !*hex || !out) return 0;
    int len = strlen(hex);
    if (len % 2 != 0) return 0;
    len /= 2;
    for (int i = 0; i < len; i++) {
        char b1, b2;
        if (!hexchr2bin(hex[i*2], &b1) || !hexchr2bin(hex[i*2+1], &b2)) return 0;
        out[i] = (b1 << 4) | b2;
    }
    return len;
}

int main() {
    D_HashRmd rmd;
    hexs2bin(rmdHex, rmd);
    start_time = time(NULL);
	
	printf("\r%s\033[K\n", "==============.....SEARCHING......==============");
	printf("\rRmd160: %s\033[K\n", rmdHex);
	printf("\r%s\033[K\n", "==============.....SEARCHING......==============");
    printf("\rElapsed: %s\033[K\n", "");
	printf("\rKecepatan: %.2f Key/s\033[K\n", (double)0);
	printf("\rPrivateKey: %s\033[K\n", "");
	printf("\rRmd160: %s\033[K\n", "");
	printf("\r%s\033[K\n", "================================================");
    fflush(stdout);

    unsigned char GR[4];
    D_Int pvk = {0};
    D_Point pbk;
    D_Result res;
    bool keyFound = false;

    while (!keyFound) {
		#ifdef TEST
			GR[0] = 0x1d;
			GR[1] = 0x83;
			GR[2] = 0x27;
			GR[3] = 0x5f;
		#else
			while (generate_random(GR, 4) != 0);
		#endif

        pvk.uc[4] = GR[3];
        pvk.uc[5] = GR[2];
        pvk.uc[6] = GR[1];
        pvk.uc[7] = GR[0];

        for (uint8_t lprefix = MINPREFIX; lprefix <= MAXPREFIX; lprefix++) {
            if (keyFound) break;

            pvk.uc[8] = lprefix;
            _CPU_ComputePublicKey(&pvk, &pbk);

            clock_t t = clock();
            _Launch_DKrek_Kernel(&pvk, &pbk, &rmd, &res, SIZET);
            t = clock() - t;

            double time_taken = ((double)t) / CLOCKS_PER_SEC;
            time_t now = time(NULL);
            double elapsed = difftime(now, start_time);

            char buffer1[19] = {0}, buffer2[41] = {0};
            for (int i = 0; i < 9; i++) sprintf(&buffer1[i*2], "%.2x", res.privateKey.uc[8 - i]);
            for (int i = 0; i < 20; i++) sprintf(&buffer2[i*2], "%.2x", res.rmd160[i]);

            printf("\033[8A");
            if (res.keyFound) {
                printf("\r%s\033[K\n", "==============.....--FOUND--......==============");
                printf("\rRmd160: %s\033[K\n", rmdHex);
                printf("\r%s\033[K\n", "==============.....--FOUND--......==============");
                printf("\rElapsed: %s\033[K\n", format_time(elapsed));
                printf("\rKecepatan: %.2f Key/s\033[K\n", KEYTOFINDCOUNT/time_taken);
                printf("\rPrivateKey: %s\033[K\n", buffer1);
                printf("\rRmd160: %s\033[K\n", buffer2);
				printf("\r%s\033[K\n", "================================================");

                FILE *f = fopen("/home/dhani/DATA/DKrek/___KEYFOUND_KEYFOUND_KEYFOUND___.txt", "a+");
                if (f) { fprintf(f, "%s\n%s\n", buffer1, buffer2); fclose(f); }
                f = fopen("/home/dhani/DATA/DKrek/money.txt", "a+");
                if (f) { fprintf(f, "%s", buffer1); fclose(f); }

                keyFound = true;
                break;
            } else {
                printf("\r%s\033[K\n", "==============.....SEARCHING......==============");
                printf("\rRmd160: %s\033[K\n", rmdHex);
                printf("\r%s\033[K\n", "==============.....SEARCHING......==============");
                printf("\rElapsed: %s\033[K\n", format_time(elapsed));
                printf("\rKecepatan: %.2f Key/s\033[K\n", KEYTOFINDCOUNT/time_taken);
                printf("\rPrivateKey: %s\033[K\n", buffer1);
                printf("\rRmd160: %s\033[K\n", buffer2);
				printf("\r%s\033[K\n", "================================================");
            }
            fflush(stdout);
        }
    }

    return 0;
}
