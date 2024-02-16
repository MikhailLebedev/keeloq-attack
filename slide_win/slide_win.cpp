#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>
#include <ctime>
#include <emmintrin.h>
#include <immintrin.h>
#include "mpi.h"

#define OPT_AVX // OPT_DEF / OPT_SSE / OPT_AVX

#define KEELOQ_NLF           0x3A5C742E
#define g5e(x)               (((x >> 1) & 1) ^ ((x >> 8) & 2) ^ ((x >> 18) & 4) ^ ((x >> 23) & 8) ^ ((x >> 27) & 16))
#define g5d(x)               (((x >> 0) & 1) ^ ((x >> 7) & 2) ^ ((x >> 17) & 4) ^ ((x >> 22) & 8) ^ ((x >> 26) & 16))
#define DICT_SIZE            0x10000 // 2^16 65536 plaintext/ciphertext pairs
#define TABLE_SIZE           0x10000 // 2^16 65536 - search collision size 16bit
#define TABLE_ROW_SIZE       16
#define KEELOQ_PART          16 // part encrypt/decrypt by 16bits of key
#define KEELOQ_FULL          528 // full encrypt/decrypt

using namespace std;

/*
cipher[i] = Encrypt(plain[i])
cipher[j] = Encrypt(plain[j])

         k0        k1            k2          k3          k0
plain[i] -> plainE    -> plainEE -- plainD   <- plain[j]
            cipher[i] -> cipherE -- cipherDD <- cipherD  <- cipher[j]
*/

uint32_t plain[DICT_SIZE];
uint32_t cipher[DICT_SIZE];
uint32_t plainE[DICT_SIZE];   // encrypt by k0 from plain
uint32_t cipherD[DICT_SIZE];  // decrypt by k0 from cipher
uint32_t k1_arr[DICT_SIZE];
uint32_t cipherE[DICT_SIZE];  // encrypt by k1 from cipher
uint32_t k3_arr[DICT_SIZE];
uint32_t cipherDD[DICT_SIZE]; // decrypt by k3 from cipherD
uint32_t plainEE[DICT_SIZE];  // encrypt by k1 from plainE
uint16_t table[TABLE_SIZE][TABLE_ROW_SIZE];

uint32_t KlEncrypt(uint32_t data, uint64_t key, int rounds)
{
    for (int r = 0; r < rounds; r++)
        data = (data >> 1) ^ (((data ^ (data >> 16) ^ (uint32_t)(key >> (r & 63)) ^ (KEELOQ_NLF >> g5e(data))) & 1) << 31);
    return data;
}

uint32_t KlDecrypt(uint32_t data, uint64_t key, int rounds)
{
    for (int r = 0; r < rounds; r++)
        data = (data << 1) ^ ((data >> 31) ^ (data >> 15) ^ (uint32_t)(key >> ((15 - r) & 63)) ^ (KEELOQ_NLF >> g5d(data)) & 1);
    return data;
}

uint16_t KlKeyPart(uint32_t data, uint16_t hint)
{
    uint16_t key = 0;
    for (int i = 0; i < KEELOQ_PART; i++)
    {
        key = key ^ (data ^ (data >> 16) ^ (KEELOQ_NLF >> g5e(data)) ^ (hint >> i)) << i;
        data = (data >> 1) ^ ((hint >> i) << 31);
    }
    return key;
}

bool KlLastKeyPart(uint32_t data1, uint16_t hint1, uint32_t data2, uint16_t hint2, uint16_t* key)
{
    uint16_t tmp1;
    uint16_t tmp2;
    uint16_t _key = 0;
    for (int i = 0; i < KEELOQ_PART; i++)
    {
        tmp1 = (data1 ^ (data1 >> 16) ^ (KEELOQ_NLF >> g5e(data1)) ^ (hint1 >> i)) & 1;
        tmp2 = (data2 ^ (data2 >> 16) ^ (KEELOQ_NLF >> g5e(data2)) ^ (hint2 >> i)) & 1;
        if (tmp1 != tmp2)
            return false;
        _key ^= (tmp1 << i);
        data1 = (data1 >> 1) ^ ((hint1 >> i) << 31);
        data2 = (data2 >> 1) ^ ((hint2 >> i) << 31);
    }
    *key = _key;
    return true;
}

#ifdef OPT_SSE

__m128i sse_mask_p0 = _mm_set1_epi32(1);
__m128i sse_mask_p1 = _mm_set1_epi32(2);
__m128i sse_mask_p2 = _mm_set1_epi32(4);
__m128i sse_mask_p3 = _mm_set1_epi32(8);
__m128i sse_mask_p4 = _mm_set1_epi32(16);
__m128i sse_nlf = _mm_set1_epi32(KEELOQ_NLF);

__m128i sse_KlEncrypt(__m128i data, __m128i key)
{
    __m128i tmp;
    __m128i shift;
    for (int r = 0; r < KEELOQ_PART; r++)
    {
        tmp = _mm_srli_epi32(key, r);
        tmp = _mm_xor_si128(tmp, data);
        tmp = _mm_xor_si128(tmp, _mm_srli_epi32(data, 16));
        shift = _mm_xor_si128(_mm_and_si128(_mm_srli_epi32(data, 1), sse_mask_p0),
            _mm_xor_si128(
                _mm_xor_si128(_mm_and_si128(_mm_srli_epi32(data, 8), sse_mask_p1),
                    _mm_and_si128(_mm_srli_epi32(data, 18), sse_mask_p2)),
                _mm_xor_si128(_mm_and_si128(_mm_srli_epi32(data, 23), sse_mask_p3),
                    _mm_and_si128(_mm_srli_epi32(data, 27), sse_mask_p4))
            )
        );
        tmp = _mm_xor_si128(tmp, _mm_srlv_epi32(sse_nlf, shift));
        data = _mm_xor_si128(_mm_slli_epi32(tmp, 31), _mm_srli_epi32(data, 1));
    }
    return data;
}

__m128i sse_KlDecrypt(__m128i data, __m128i key)
{
    __m128i tmp;
    __m128i shift;
    for (int r = 0; r < KEELOQ_PART; r++)
    {
        tmp = _mm_srli_epi32(key, 15 - r);
        tmp = _mm_xor_si128(tmp, _mm_srli_epi32(data, 31));
        tmp = _mm_xor_si128(tmp, _mm_srli_epi32(data, 15));
        shift = _mm_xor_si128(_mm_and_si128(_mm_srli_epi32(data, 0), sse_mask_p0),
            _mm_xor_si128(
                _mm_xor_si128(_mm_and_si128(_mm_srli_epi32(data, 7), sse_mask_p1),
                    _mm_and_si128(_mm_srli_epi32(data, 17), sse_mask_p2)),
                _mm_xor_si128(_mm_and_si128(_mm_srli_epi32(data, 22), sse_mask_p3),
                    _mm_and_si128(_mm_srli_epi32(data, 26), sse_mask_p4))
            )
        );
        tmp = _mm_xor_si128(tmp, _mm_srlv_epi32(sse_nlf, shift));
        data = _mm_xor_si128(_mm_and_si128(tmp, sse_mask_p0), _mm_slli_epi32(data, 1));
    }
    return data;
}

__m128i sse_KlKeyPart(__m128i data, __m128i hint)
{
    __m128i tmp;
    __m128i shift;
    __m128i key = _mm_set1_epi32(0);
    for (int r = 0; r < KEELOQ_PART; r++)
    {
        tmp = _mm_xor_si128(data, _mm_srli_epi32(data, 16));
        tmp = _mm_xor_si128(tmp, _mm_srli_epi32(hint, r));
        shift = _mm_xor_si128(_mm_and_si128(_mm_srli_epi32(data, 1), sse_mask_p0),
            _mm_xor_si128(
                _mm_xor_si128(_mm_and_si128(_mm_srli_epi32(data, 8), sse_mask_p1),
                    _mm_and_si128(_mm_srli_epi32(data, 18), sse_mask_p2)),
                _mm_xor_si128(_mm_and_si128(_mm_srli_epi32(data, 23), sse_mask_p3),
                    _mm_and_si128(_mm_srli_epi32(data, 27), sse_mask_p4))
            )
        );
        tmp = _mm_xor_si128(tmp, _mm_srlv_epi32(sse_nlf, shift));
        tmp = _mm_slli_epi32(_mm_and_si128(tmp, sse_mask_p0), r);
        key = _mm_xor_si128(key, tmp);
        data = _mm_xor_si128(_mm_srli_epi32(data, 1), _mm_slli_epi32(_mm_srli_epi32(hint, r), 31));
    }
    return key;
}

#endif

#ifdef OPT_AVX

__m256i avx_mask_p0 = _mm256_set1_epi32(1);
__m256i avx_mask_p1 = _mm256_set1_epi32(2);
__m256i avx_mask_p2 = _mm256_set1_epi32(4);
__m256i avx_mask_p3 = _mm256_set1_epi32(8);
__m256i avx_mask_p4 = _mm256_set1_epi32(16);
__m256i avx_nlf = _mm256_set1_epi32(KEELOQ_NLF);

__m256i avx_KlEncrypt(__m256i data, __m256i key)
{
    __m256i tmp;
    __m256i shift;
    for (int r = 0; r < KEELOQ_PART; r++)
    {
        tmp = _mm256_srli_epi32(key, r);
        tmp = _mm256_xor_si256(tmp, data);
        tmp = _mm256_xor_si256(tmp, _mm256_srli_epi32(data, 16));
        shift = _mm256_xor_si256(_mm256_and_si256(_mm256_srli_epi32(data, 1), avx_mask_p0),
            _mm256_xor_si256(
                _mm256_xor_si256(_mm256_and_si256(_mm256_srli_epi32(data, 8), avx_mask_p1),
                    _mm256_and_si256(_mm256_srli_epi32(data, 18), avx_mask_p2)),
                _mm256_xor_si256(_mm256_and_si256(_mm256_srli_epi32(data, 23), avx_mask_p3),
                    _mm256_and_si256(_mm256_srli_epi32(data, 27), avx_mask_p4))
            )
        );
        tmp = _mm256_xor_si256(tmp, _mm256_srlv_epi32(avx_nlf, shift));
        data = _mm256_xor_si256(_mm256_slli_epi32(tmp, 31), _mm256_srli_epi32(data, 1));
    }
    return data;
}

__m256i avx_KlDecrypt(__m256i data, __m256i key)
{
    __m256i tmp;
    __m256i shift;
    for (int r = 0; r < KEELOQ_PART; r++)
    {
        tmp = _mm256_srli_epi32(key, 15 - r);
        tmp = _mm256_xor_si256(tmp, _mm256_srli_epi32(data, 31));
        tmp = _mm256_xor_si256(tmp, _mm256_srli_epi32(data, 15));
        shift = _mm256_xor_si256(_mm256_and_si256(_mm256_srli_epi32(data, 0), avx_mask_p0),
            _mm256_xor_si256(
                _mm256_xor_si256(_mm256_and_si256(_mm256_srli_epi32(data, 7), avx_mask_p1),
                    _mm256_and_si256(_mm256_srli_epi32(data, 17), avx_mask_p2)),
                _mm256_xor_si256(_mm256_and_si256(_mm256_srli_epi32(data, 22), avx_mask_p3),
                    _mm256_and_si256(_mm256_srli_epi32(data, 26), avx_mask_p4))
            )
        );
        tmp = _mm256_xor_si256(tmp, _mm256_srlv_epi32(avx_nlf, shift));
        data = _mm256_xor_si256(_mm256_and_si256(tmp, avx_mask_p0), _mm256_slli_epi32(data, 1));
    }
    return data;
}

__m256i avx_KlKeyPart(__m256i data, __m256i hint)
{
    __m256i tmp;
    __m256i shift;
    __m256i key = _mm256_set1_epi32(0);
    for (int r = 0; r < KEELOQ_PART; r++)
    {
        tmp = _mm256_xor_si256(data, _mm256_srli_epi32(data, 16));
        tmp = _mm256_xor_si256(tmp, _mm256_srli_epi32(hint, r));
        shift = _mm256_xor_si256(_mm256_and_si256(_mm256_srli_epi32(data, 1), avx_mask_p0),
            _mm256_xor_si256(
                _mm256_xor_si256(_mm256_and_si256(_mm256_srli_epi32(data, 8), avx_mask_p1),
                    _mm256_and_si256(_mm256_srli_epi32(data, 18), avx_mask_p2)),
                _mm256_xor_si256(_mm256_and_si256(_mm256_srli_epi32(data, 23), avx_mask_p3),
                    _mm256_and_si256(_mm256_srli_epi32(data, 27), avx_mask_p4))
            )
        );
        tmp = _mm256_xor_si256(tmp, _mm256_srlv_epi32(avx_nlf, shift));
        tmp = _mm256_slli_epi32(_mm256_and_si256(tmp, avx_mask_p0), r);
        key = _mm256_xor_si256(key, tmp);
        data = _mm256_xor_si256(_mm256_srli_epi32(data, 1), _mm256_slli_epi32(_mm256_srli_epi32(hint, r), 31));
    }
    return key;
}

#endif

#ifdef OPT_AVX2

__m512i avx2_mask_p0 = _mm512_set1_epi32(1);
__m512i avx2_mask_p1 = _mm512_set1_epi32(2);
__m512i avx2_mask_p2 = _mm512_set1_epi32(4);
__m512i avx2_mask_p3 = _mm512_set1_epi32(8);
__m512i avx2_mask_p4 = _mm512_set1_epi32(16);
__m512i avx2_nlf = _mm512_set1_epi32(KEELOQ_NLF);

__m512i avx2_KlEncrypt(__m512i data, __m512i key)
{
    __m512i tmp;
    __m512i shift;
    for (int r = 0; r < KEELOQ_PART; r++)
    {
        tmp = _mm512_srli_epi32(key, r);
        tmp = _mm512_xor_si512(tmp, data);
        tmp = _mm512_xor_si512(tmp, _mm512_srli_epi32(data, 16));
        shift = _mm512_xor_si512(_mm512_and_si512(_mm512_srli_epi32(data, 1), avx2_mask_p0),
            _mm512_xor_si512(
                _mm512_xor_si512(_mm512_and_si512(_mm512_srli_epi32(data, 8), avx2_mask_p1),
                    _mm512_and_si512(_mm512_srli_epi32(data, 18), avx2_mask_p2)),
                _mm512_xor_si512(_mm512_and_si512(_mm512_srli_epi32(data, 23), avx2_mask_p3),
                    _mm512_and_si512(_mm512_srli_epi32(data, 27), avx2_mask_p4))
            )
        );
        tmp = _mm512_xor_si512(tmp, _mm512_srlv_epi32(avx2_nlf, shift));
        data = _mm512_xor_si512(_mm512_slli_epi32(tmp, 31), _mm512_srli_epi32(data, 1));
    }
    return data;
}

__m512i avx2_KlDecrypt(__m512i data, __m512i key)
{
    __m512i tmp;
    __m512i shift;
    for (int r = 0; r < KEELOQ_PART; r++)
    {
        tmp = _mm512_srli_epi32(key, 15 - r);
        tmp = _mm512_xor_si512(tmp, _mm512_srli_epi32(data, 31));
        tmp = _mm512_xor_si512(tmp, _mm512_srli_epi32(data, 15));
        shift = _mm512_xor_si512(_mm512_and_si512(_mm512_srli_epi32(data, 0), avx2_mask_p0),
            _mm512_xor_si512(
                _mm512_xor_si512(_mm512_and_si512(_mm512_srli_epi32(data, 7), avx2_mask_p1),
                    _mm512_and_si512(_mm512_srli_epi32(data, 17), avx2_mask_p2)),
                _mm512_xor_si512(_mm512_and_si512(_mm512_srli_epi32(data, 22), avx2_mask_p3),
                    _mm512_and_si512(_mm512_srli_epi32(data, 26), avx2_mask_p4))
            )
        );
        tmp = _mm512_xor_si512(tmp, _mm512_srlv_epi32(avx2_nlf, shift));
        data = _mm512_xor_si512(_mm512_and_si512(tmp, avx2_mask_p0), _mm512_slli_epi32(data, 1));
    }
    return data;
}

__m512i avx2_KlKeyPart(__m512i data, __m512i hint)
{
    __m512i tmp;
    __m512i shift;
    __m512i key = _mm512_set1_epi32(0);
    for (int r = 0; r < KEELOQ_PART; r++)
    {
        tmp = _mm512_xor_si512(data, _mm512_srli_epi32(data, 16));
        tmp = _mm512_xor_si512(tmp, _mm512_srli_epi32(hint, r));
        shift = _mm512_xor_si512(_mm512_and_si512(_mm512_srli_epi32(data, 1), avx2_mask_p0),
            _mm512_xor_si512(
                _mm512_xor_si512(_mm512_and_si512(_mm512_srli_epi32(data, 8), avx2_mask_p1),
                    _mm512_and_si512(_mm512_srli_epi32(data, 18), avx2_mask_p2)),
                _mm512_xor_si512(_mm512_and_si512(_mm512_srli_epi32(data, 23), avx2_mask_p3),
                    _mm512_and_si512(_mm512_srli_epi32(data, 27), avx2_mask_p4))
            )
        );
        tmp = _mm512_xor_si512(tmp, _mm512_srlv_epi32(avx2_nlf, shift));
        tmp = _mm512_slli_epi32(_mm512_and_si512(tmp, avx2_mask_p0), r);
        key = _mm512_xor_si512(key, tmp);
        data = _mm512_xor_si512(_mm512_srli_epi32(data, 1), _mm512_slli_epi32(_mm512_srli_epi32(hint, r), 31));
    }
    return key;
}

#endif

void fillPeCd(uint32_t k0)
{
#ifdef OPT_DEF
    for (int i = 0; i < DICT_SIZE; i += 1)
    {
        plainE[i] = KlEncrypt(plain[i], k0, KEELOQ_PART);
        cipherD[i] = KlDecrypt(cipher[i], k0, KEELOQ_PART);
    }
#endif
#ifdef OPT_SSE
    for (int i = 0; i < DICT_SIZE; i += 4)
    {
        *(__m128i*)&plainE[i] = sse_KlEncrypt(*(__m128i*)&plain[i], _mm_set1_epi32(k0));
        *(__m128i*)&cipherD[i] = sse_KlDecrypt(*(__m128i*)&cipher[i], _mm_set1_epi32(k0));
    }
#endif 
#ifdef OPT_AVX
    for (int i = 0; i < DICT_SIZE; i += 8)
    {
        *(__m256i*)& plainE[i] = avx_KlEncrypt(*(__m256i*) & plain[i], _mm256_set1_epi32(k0));
        *(__m256i*)& cipherD[i] = avx_KlDecrypt(*(__m256i*) & cipher[i], _mm256_set1_epi32(k0));
    }
#endif
#ifdef OPT_AVX2
    for (int i = 0; i < DICT_SIZE; i += 16)
    {
        *(__m512i*)& plainE[i] = avx2_KlEncrypt(*(__m512i*) & plain[i], _mm512_set1_epi32(k0));
        *(__m512i*)& cipherD[i] = avx2_KlDecrypt(*(__m512i*) & cipher[i], _mm512_set1_epi32(k0));
    }
#endif
}

void fillK3Cdd(uint32_t ps)
{
#ifdef OPT_DEF
    for (int i = 0; i < DICT_SIZE; i += 1)
    {
        k3_arr[i] = KlKeyPart((plain[i] << 16) ^ ps, (uint16_t)(plain[i] >> 16));
        cipherDD[i] = KlDecrypt(cipherD[i], k3_arr[i], KEELOQ_PART);
    }
#endif
#ifdef OPT_SSE
    for (int i = 0; i < DICT_SIZE; i += 4)
    {
        *(__m128i*)& k3_arr[i] = sse_KlKeyPart(_mm_xor_si128(_mm_slli_epi32(*(__m128i*) & plain[i], 16), _mm_set1_epi32(ps)), _mm_srli_epi32(*(__m128i*) & plain[i], 16));
        *(__m128i*)& cipherDD[i] = sse_KlDecrypt(*(__m128i*) & cipherD[i], *(__m128i*) & k3_arr[i]);
    }
#endif 
#ifdef OPT_AVX
    for (int i = 0; i < DICT_SIZE; i += 8)
    {
        *(__m256i*)& k3_arr[i] = avx_KlKeyPart(_mm256_xor_si256(_mm256_slli_epi32(*(__m256i*) & plain[i], 16), _mm256_set1_epi32(ps)), _mm256_srli_epi32(*(__m256i*) & plain[i], 16));
        *(__m256i*)& cipherDD[i] = avx_KlDecrypt(*(__m256i*) & cipherD[i], *(__m256i*) & k3_arr[i]);
    }
#endif
#ifdef OPT_AVX2
    for (int i = 0; i < DICT_SIZE; i += 16)
    {
        *(__m512i*)& k3_arr[i] = avx2_KlKeyPart(_mm512_xor_si512(_mm512_slli_epi32(*(__m512i*) & plain[i], 16), _mm512_set1_epi32(ps)), _mm512_srli_epi32(*(__m512i*) & plain[i], 16));
        *(__m512i*)& cipherDD[i] = avx2_KlDecrypt(*(__m512i*) & cipherD[i], *(__m512i*) & k3_arr[i]);
    }
#endif
}

void fillK1Ce(uint32_t ps)
{
#ifdef OPT_DEF
    for (int i = 0; i < DICT_SIZE; i += 1)
    {
        k1_arr[i] = KlKeyPart(plainE[i], ps);
        cipherE[i] = KlEncrypt(cipher[i], k1_arr[i], KEELOQ_PART);
    }
#endif
#ifdef OPT_SSE
    for (int i = 0; i < DICT_SIZE; i += 4)
    {
        *(__m128i*)& k1_arr[i] = sse_KlKeyPart(*(__m128i*) & plainE[i], _mm_set1_epi32(ps));
        *(__m128i*)& cipherE[i] = sse_KlEncrypt(*(__m128i*) & cipher[i], *(__m128i*) & k1_arr[i]);
    }
#endif 
#ifdef OPT_AVX
    for (int i = 0; i < DICT_SIZE; i += 8)
    {
        *(__m256i*)& k1_arr[i] = avx_KlKeyPart(*(__m256i*) & plainE[i], _mm256_set1_epi32(ps));
        *(__m256i*)& cipherE[i] = avx_KlEncrypt(*(__m256i*) & cipher[i], *(__m256i*) & k1_arr[i]);
    }
#endif
#ifdef OPT_AVX2
    for (int i = 0; i < DICT_SIZE; i += 16)
    {
        *(__m512i*)& k1_arr[i] = avx2_KlKeyPart(*(__m512i*) & plainE[i], _mm512_set1_epi32(ps));
        *(__m512i*)& cipherE[i] = avx2_KlEncrypt(*(__m512i*) & cipher[i], *(__m512i*) & k1_arr[i]);
    }
#endif
}

void fillK3CddK1CePee(uint32_t ps)
{
#ifdef OPT_DEF
    for (int i = 0; i < DICT_SIZE; i += 1)
    {
        k3_arr[i] = KlKeyPart((plain[i] << 16) ^ ps, (uint16_t)(plain[i] >> 16));
        cipherDD[i] = KlDecrypt(cipherD[i], k3_arr[i], KEELOQ_PART);
        k1_arr[i] = KlKeyPart(plainE[i], ps);
        cipherE[i] = KlEncrypt(cipher[i], k1_arr[i], KEELOQ_PART);
        plainEE[i] = (plainE[i] >> 16) ^ (ps << 16);
    }
#endif
#ifdef OPT_SSE
    for (int i = 0; i < DICT_SIZE; i += 4)
    {
        *(__m128i*)& k3_arr[i] = sse_KlKeyPart(_mm_xor_si128(_mm_slli_epi32(*(__m128i*) & plain[i], 16), _mm_set1_epi32(ps)), _mm_srli_epi32(*(__m128i*) & plain[i], 16));
        *(__m128i*)& cipherDD[i] = sse_KlDecrypt(*(__m128i*) & cipherD[i], *(__m128i*) & k3_arr[i]);
        *(__m128i*)& k1_arr[i] = sse_KlKeyPart(*(__m128i*) & plainE[i], _mm_set1_epi32(ps));
        *(__m128i*)& cipherE[i] = sse_KlEncrypt(*(__m128i*) & cipher[i], *(__m128i*) & k1_arr[i]);
        *(__m128i*)& plainEE[i] = _mm128_xor_si128(_mm128_srli_epi32(*(__m128i*) & plainE[i], 16), _mm128_slli_epi32(_mm128_set1_epi32(ps), 16));
    }
#endif 
#ifdef OPT_AVX
    for (int i = 0; i < DICT_SIZE; i += 8)
    {
        *(__m256i*)& k3_arr[i] = avx_KlKeyPart(_mm256_xor_si256(_mm256_slli_epi32(*(__m256i*) & plain[i], 16), _mm256_set1_epi32(ps)), _mm256_srli_epi32(*(__m256i*) & plain[i], 16));
        *(__m256i*)& cipherDD[i] = avx_KlDecrypt(*(__m256i*) & cipherD[i], *(__m256i*) & k3_arr[i]);
        *(__m256i*)& k1_arr[i] = avx_KlKeyPart(*(__m256i*) & plainE[i], _mm256_set1_epi32(ps));
        *(__m256i*)& cipherE[i] = avx_KlEncrypt(*(__m256i*) & cipher[i], *(__m256i*) & k1_arr[i]);
        *(__m256i*)& plainEE[i] = _mm256_xor_si256(_mm256_srli_epi32(*(__m256i*) & plainE[i], 16), _mm256_slli_epi32(_mm256_set1_epi32(ps), 16));
    }
#endif
#ifdef OPT_AVX2
    for (int i = 0; i < DICT_SIZE; i += 16)
    {
        *(__m512i*)&k3_arr[i] = avx2_KlKeyPart(_mm512_xor_si512(_mm512_slli_epi32(*(__m512i*)&plain[i], 16), _mm512_set1_epi32(ps)), _mm512_srli_epi32(*(__m512i*)&plain[i], 16));
        *(__m512i*)&cipherDD[i] = avx2_KlDecrypt(*(__m512i*)&cipherD[i], *(__m512i*)&k3_arr[i]);
        *(__m512i*)&k1_arr[i] = avx2_KlKeyPart(*(__m512i*)&plainE[i], _mm512_set1_epi32(ps));
        *(__m512i*)&cipherE[i] = avx2_KlEncrypt(*(__m512i*)&cipher[i], *(__m512i*)&k1_arr[i]);
        *(__m512i*)& plainEE[i] = _mm512_xor_si512(_mm512_srli_epi32(*(__m512i*) & plainE[i], 16), _mm512_slli_epi32(_mm512_set1_epi32(ps), 16));

    }
#endif
}

void zeroTable()
{
    for (int i = 0; i < TABLE_SIZE; i++)
    {
        table[i][0] = 0;
    }
}

void fillTable()
{
    for (int i = 0; i < DICT_SIZE; i++)
    {
        uint16_t lo = cipherDD[i] & 0xFFFF;
        table[lo][(table[lo][0] % (TABLE_ROW_SIZE - 1)) + 1] = i;
        table[lo][0]++;
        if (table[lo][0] >= TABLE_ROW_SIZE)
        {
            cout << "Mem" << endl;
        }
    }
}

bool finalCheck(uint64_t key)
{
    if (cipher[0] == KlEncrypt(plain[0], key, KEELOQ_FULL))
        return true;
    return false;
}

void searchPair(uint16_t k0, uint16_t ps)
{
    for (int i = 0; i < DICT_SIZE; i++)
    {
        const uint16_t searchValue = cipherE[i] >> 16;
        const uint16_t* ptr = &table[searchValue][0];
        const int amt = ptr[0] & 0xFF;
        for (int k = 1; k <= amt; k++)
        {
            int j = ptr[k];
            uint16_t k2;
            if (KlLastKeyPart(plainEE[i], plain[j], cipherE[i], cipherD[j], &k2))
            {
                uint64_t key = (((uint64_t)k3_arr[j]) << 48) ^ ((uint64_t)k2 << 32) ^ ((uint64_t)k1_arr[i] << 16) ^ k0;
                if (finalCheck(key))
                {
                    cout << "Success! Key: " << hex << key << endl;
                }
            }
        }
    }
}

int main(int argc, char** argv)
{
    ifstream inFile("./test_data_set.txt", fstream::in);
    for (int i = 0; i < DICT_SIZE; i++)
        inFile >> hex >> plain[i] >> cipher[i];

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    double startTime, endTime;
    int start = stoi(argv[1], nullptr, 16);
    int end = stoi(argv[2], nullptr, 16);

    for (uint32_t k0 = start; k0 < end; k0++)
    {
        startTime = MPI_Wtime();
        fillPeCd(k0);
        for (uint32_t ps = 0x1000 / size * rank; ps < 0x1000 / size * (rank + 1); ps++)
        {
            fillK3CddK1CePee(ps);
            zeroTable();
            fillTable();
            searchPair(k0, ps);
        }
        endTime = MPI_Wtime();
        cout << "Process: " << rank << ", k0: " << hex << k0 << dec << ", time: " << endTime - startTime << endl;
    }
    MPI_Finalize();

    return 0;
}