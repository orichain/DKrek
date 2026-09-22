#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <immintrin.h>
#include <sys/random.h>
#include <errno.h>
#include <sqlite3.h>

#include <cuda_runtime.h>
#include "kernel.h"
#include "CPUSECP256K1.h"
#include "sha256.h"
#include "base58.h"

int seeds_open(sqlite3 **db, const char *filename) {
    int rc;
    rc = sqlite3_open(filename, db);
    if (rc != SQLITE_OK) {
        if (*db) {
            sqlite3_close(*db);
            *db = NULL;
        }
        return rc;
    }
    rc = sqlite3_exec(*db, "CREATE TABLE IF NOT EXISTS seeds (seed INTEGER NOT NULL PRIMARY KEY, cnt INTEGER NOT NULL, islast INTEGER NOT NULL CHECK (islast IN (0, 1)));", NULL, NULL, NULL);
    if (rc != SQLITE_OK) {
        sqlite3_close(*db);
        *db = NULL;
        return rc;
    }
    return SQLITE_OK;
}


void seeds_close(sqlite3 *db) {
    if (db) sqlite3_close(db);
}

int seeds_append(sqlite3 *db, uint32_t seed, uint32_t cnt) {
    int rc;
    rc = sqlite3_exec(db, "BEGIN TRANSACTION;", NULL, NULL, NULL);
    if (rc != SQLITE_OK) return rc;
    rc = sqlite3_exec(db, "UPDATE seeds SET islast = 0;", NULL, NULL, NULL);
    if (rc != SQLITE_OK) goto rollback;
    sqlite3_stmt *stmt;
    rc = sqlite3_prepare_v2(db, "INSERT INTO seeds (seed, cnt, islast) VALUES (?, ?, 1);", -1, &stmt, NULL);
    if (rc != SQLITE_OK) goto rollback;
    sqlite3_bind_int64(stmt, 1, (sqlite3_int64)seed);
    sqlite3_bind_int64(stmt, 2, (sqlite3_int64)cnt);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE) goto rollback;
    return sqlite3_exec(db, "COMMIT;", NULL, NULL, NULL);
rollback:
    sqlite3_exec(db, "ROLLBACK;", NULL, NULL, NULL);
    return rc;
}


int seeds_update_cnt(sqlite3 *db, uint32_t seed, uint32_t cnt) {
    sqlite3_stmt *stmt;
    int rc;
    rc = sqlite3_prepare_v2(db, "UPDATE seeds SET cnt = ? WHERE seed = ?;", -1, &stmt, NULL);
    if (rc != SQLITE_OK) return rc;
    sqlite3_bind_int64(stmt, 1, (sqlite3_int64)cnt);
    sqlite3_bind_int64(stmt, 2, (sqlite3_int64)seed);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return (rc == SQLITE_DONE) ? SQLITE_OK : rc;
}


int seeds_get_last(sqlite3 *db, uint32_t *seed, uint32_t *cnt) {
    sqlite3_stmt *stmt;
    int rc;
    rc = sqlite3_prepare_v2(db, "SELECT seed, cnt FROM seeds WHERE islast = 1 LIMIT 1;", -1, &stmt, NULL);
    if (rc != SQLITE_OK) return rc;
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
        *seed = (uint32_t)sqlite3_column_int64(stmt, 0);
        *cnt  = (uint32_t)sqlite3_column_int64(stmt, 1);
        sqlite3_finalize(stmt);
        return SQLITE_OK;
    }
    sqlite3_finalize(stmt);
    return (rc == SQLITE_DONE) ? SQLITE_NOTFOUND : rc;
}

int seeds_get_cnt(sqlite3 *db, uint32_t seed, uint32_t *cnt) {
    sqlite3_stmt *stmt;
    int rc;
    rc = sqlite3_prepare_v2(db, "SELECT cnt FROM seeds WHERE seed = ? LIMIT 1;", -1, &stmt, NULL);
    if (rc != SQLITE_OK) return rc;
    sqlite3_bind_int64(stmt, 1, (sqlite3_int64)seed);
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
        *cnt = (uint32_t)sqlite3_column_int64(stmt, 0);
        sqlite3_finalize(stmt);
        return SQLITE_OK;
    }
    sqlite3_finalize(stmt);
    return (rc == SQLITE_DONE) ? SQLITE_NOTFOUND : rc;
}

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

inline static void PPRNG32_seed(PPRNG32* rand, uint32_t seed) {
	rand->seed = seed;
}

uint32_t PPRNG32_rand(PPRNG32* rand, uint8_t rng_count) {
    uint32_t y = (rand->seed ^ rng_count);
    y ^= y >> 16;
    y *= 0x7feb352dU;
    y ^= y >> 15;
    y *= 0x846ca68bU;
    y ^= y >> 16;

    return y;
}

int new_seed(sqlite3 *db, PPRNG32 *r) {
	uint32_t seed;
	uint32_t cnt = 0;
	int rc = SQLITE_OK;
	do {
        if (__builtin_cpu_supports("rdseed")) {
            int retries = 10;
            while (retries-- > 0 && !_rdseed32_step(&seed));
            if (retries < 0) {
                ssize_t ret;
                do {
                    ret = getrandom(&seed, sizeof(seed), 0);
                } while (ret < 0 && errno == EINTR);
                if (ret != sizeof(seed)) {
                    return -1;
                }
            }
        } else {
            ssize_t ret;
            do {
                ret = getrandom(&seed, sizeof(seed), 0);
            } while (ret < 0 && errno == EINTR);
            if (ret != sizeof(seed)) {
                return -1;
            }
        }
		rc = seeds_get_cnt(db, seed, &cnt);
	} while (rc == SQLITE_OK);
	PPRNG32_seed(r, seed);
	r->seeded = 0x01;
	r->rng_count = 0;
	r->seed = seed;
	seeds_append(db, seed, 0);
	return 0;
}

int generate_random(sqlite3 *db, PPRNG32 *r, uint8_t *buffer) {
	if (!r) return -1;
    if (!buffer) return -1;
    if (r->seeded == 0x00) {
		uint32_t cseed = 0;
		uint32_t ccnt = 0;
		int rc = seeds_get_last(db, &cseed, &ccnt);
		if (rc == SQLITE_OK) {
			if (ccnt >= RESEEDCNT) {
				if (new_seed(db, r) == -1) return -1;
			} else {
				PPRNG32_seed(r, cseed);
				r->seeded = 0x01;
				r->seed = cseed;
				r->rng_count = ccnt;
			}
		} else if (rc == SQLITE_NOTFOUND) {
			if (new_seed(db, r) == -1) return -1;
		}
	} else {
		if (r->rng_count >= RESEEDCNT) {
			if (new_seed(db, r) == -1) return -1;
		}
	}
	uint32_t rnd = PPRNG32_rand(r, r->rng_count);
	uint32_t rnd_be32 = htobe32(rnd);
	memcpy(buffer, &rnd_be32, sizeof(uint32_t));
	r->rng_count++;
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

static bool b58_sha256(
    void *out,
    const void *data,
    size_t len)
{
    sha256(
        (uint8_t *)data,
        len,
        (uint8_t *)out
    );

    return true;
}


bool privatekey_to_wif(
    char *wif,
    size_t *wif_sz,
    const uint8_t privatekey[32],
    bool compressed)
{
    uint8_t data[33];
    size_t datasz;

    /*
     * Private key = 32 byte
     */
    memcpy(data, privatekey, 32);

    /*
     * Compressed WIF:
     * append 0x01
     */
    if (compressed) {
        data[32] = 0x01;
        datasz = 33;
    } else {
        datasz = 32;
    }

    /*
     * Bitcoin mainnet WIF version
     */
    return b58check_enc(
        wif,
        wif_sz,
        0x80,
        data,
        datasz
    );
}

int main() {
	sqlite3 *db;
	if (seeds_open(&db, dbName) != SQLITE_OK) {
		printf("Failed to open database\n");
		return 1;
	}
	
	time_t start_time;
    D_HashRmd rmd;
    PPRNG32 r;
    r.seed = 0;
    r.seeded = 0x00;
    r.rng_count = 0;
    
    hexs2bin(rmdHex, rmd);
    start_time = time(NULL);
	
	printf("\r%s\033[K\n", "==============.....SEARCHING......==============");
	printf("\rRmd160: %s\033[K\n", rmdHex);
	printf("\r%s\033[K\n", "==============.....SEARCHING......==============");
    printf("\rElapsed: %s\033[K\n", "");
	printf("\rSpeed: %.2f Key/s\033[K\n", (double)0);
	printf("\rRng Cnt: %d\033[K\n", r.rng_count);
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
			while (generate_random(db, &r, GR) != 0);
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
            printf("\033[9A");
            if (res.keyFound) {
                printf("\r%s\033[K\n", "==============.....--FOUND--......==============");
                printf("\rRmd160: %s\033[K\n", rmdHex);
                printf("\r%s\033[K\n", "==============.....--FOUND--......==============");
                printf("\rElapsed: %s\033[K\n", format_time(elapsed));
                printf("\rSpeed: %.2f Key/s\033[K\n", KEYTOFINDCOUNT/time_taken);
                printf("\rRng Cnt: %d\033[K\n", r.rng_count);
                printf("\rPrivateKey: %s\033[K\n", buffer1);
                printf("\rRmd160: %s\033[K\n", buffer2);
				printf("\r%s\033[K\n", "================================================");

                FILE *f = fopen("money.txt", "a+");
                if (f) { fprintf(f, "%s\n%s\n", buffer1, buffer2); fclose(f); }
                
                uint8_t cprivatekey[32];
                for (int i = 0 ; i < 32 ; i++) {
                    cprivatekey[i] = res.privateKey.uc[31 - i];
                }
                char wif[64];
                size_t wif_sz = sizeof(wif) - 1;
                b58_sha256_impl = b58_sha256;
                privatekey_to_wif(wif, &wif_sz, cprivatekey, true);
                wif[wif_sz] = '\0';
                
                f = fopen("money_wif.txt", "a+");
                if (f) { fprintf(f, "%s\n", wif); fclose(f); }

                keyFound = true;
                break;
            } else {
                printf("\r%s\033[K\n", "==============.....SEARCHING......==============");
                printf("\rRmd160: %s\033[K\n", rmdHex);
                printf("\r%s\033[K\n", "==============.....SEARCHING......==============");
                printf("\rElapsed: %s\033[K\n", format_time(elapsed));
                printf("\rSpeed: %.2f Key/s\033[K\n", KEYTOFINDCOUNT/time_taken);
                printf("\rRng Cnt: %d\033[K\n", r.rng_count);
                printf("\rPrivateKey: %s\033[K\n", buffer1);
                printf("\rRmd160: %s\033[K\n", buffer2);
				printf("\r%s\033[K\n", "================================================");
            }
            fflush(stdout);
        }
        seeds_update_cnt(db, r.seed, r.rng_count);
    }
    
    seeds_close(db);

    return 0;
}
