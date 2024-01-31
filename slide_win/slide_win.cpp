#include <iostream>
#include <fstream>
#include <thread>
#include <ctime>
#include <format>
#include <emmintrin.h>

#define THREADS 16

#define KEELOQ_NLF              0x3A5C742E
#define bit(x, n)               (((x) >> (n))&1)
#define g5(x, a, b, c, d, e)    (bit(x, a) + bit(x, b) * 2 + bit(x, c) * 4 + bit(x, d) * 8 + bit(x, e) * 16)
#define g5e(x)                  (((x >> 1) & 1) ^ ((x >> 8) & 2) ^ ((x >> 18) & 4) ^ ((x >> 23) & 8) ^ ((x >> 27) & 16))
#define g5d(x)                  (((x >> 0) & 1) ^ ((x >> 7) & 2) ^ ((x >> 17) & 4) ^ ((x >> 22) & 8) ^ ((x >> 26) & 16))

#define DICT_SIZE 0x10000
#define TABLE_SIZE 16

#define KEELOQ_PART 16
#define KEELOQ_FULL 528

uint32_t plain[DICT_SIZE];
uint32_t cipher[DICT_SIZE];
thread_local uint32_t plainE[DICT_SIZE];
thread_local uint32_t plainD[DICT_SIZE];

thread_local uint32_t cipherD[DICT_SIZE];
thread_local uint32_t cipherDD[DICT_SIZE];
thread_local uint32_t cipherE[DICT_SIZE];

thread_local uint32_t k1_arr[DICT_SIZE];
thread_local uint32_t k3_arr[DICT_SIZE];

thread_local uint16_t table[DICT_SIZE][TABLE_SIZE];

bool flag = true;

using namespace std;

uint32_t KlEncrypt(uint32_t data, uint64_t key, int n)
{
    for (int r = 0; r < n; r++)
        data = (data >> 1) ^ (((data ^ (data >> 16) ^ (uint32_t)(key >> (r & 63)) ^ (KEELOQ_NLF >> g5e(data))) & 1) << 31);
    return data;
}

uint32_t KlDecrypt(uint32_t data, uint64_t key, int n)
{
    for (int r = 0; r < n; r++)
        data = (data << 1) ^ bit(data, 31) ^ bit(data, 15) ^ (uint32_t)bit(key, (15 - r) & 63) ^ bit(KEELOQ_NLF, g5(data, 0, 8, 19, 25, 30));
    return data;
}

__m128i sse_KlEncrypt(__m128i data, __m128i key, int n)
{
    __m128i tmp;
    uint32_t arr[4];
    __m128i sse_mask_p0 = _mm_set1_epi32(1);
    __m128i sse_mask_p1 = _mm_set1_epi32(2);
    __m128i sse_mask_p2 = _mm_set1_epi32(4);
    __m128i sse_mask_p3 = _mm_set1_epi32(8);
    __m128i sse_mask_p4 = _mm_set1_epi32(16);
    for (int r = 0; r < n; r++)
    {
        tmp = _mm_srli_epi32(key, r);
        tmp = _mm_xor_si128(tmp, data);
        tmp = _mm_xor_si128(tmp, _mm_srli_epi32(data, 16));
        *(__m128i*)arr = _mm_xor_si128(_mm_and_si128(_mm_srli_epi32(data, 1), sse_mask_p0),
            _mm_xor_si128(
                _mm_xor_si128(_mm_and_si128(_mm_srli_epi32(data, 8), sse_mask_p1),
                    _mm_and_si128(_mm_srli_epi32(data, 18), sse_mask_p2)),
                _mm_xor_si128(_mm_and_si128(_mm_srli_epi32(data, 23), sse_mask_p3),
                    _mm_and_si128(_mm_srli_epi32(data, 27), sse_mask_p4))
            )
        );
        tmp = _mm_xor_si128(tmp, _mm_set_epi32(KEELOQ_NLF >> arr[3], KEELOQ_NLF >> arr[2], KEELOQ_NLF >> arr[1], KEELOQ_NLF >> arr[0]));
        data = _mm_xor_si128(_mm_slli_epi32(tmp, 31), _mm_srli_epi32(data, 1));
    }
    return data;
}

__m128i sse_KlDecrypt(__m128i data, __m128i key, int n)
{
    __m128i tmp;
    uint32_t arr[4];
    __m128i sse_mask_p0 = _mm_set1_epi32(1);
    __m128i sse_mask_p1 = _mm_set1_epi32(2);
    __m128i sse_mask_p2 = _mm_set1_epi32(4);
    __m128i sse_mask_p3 = _mm_set1_epi32(8);
    __m128i sse_mask_p4 = _mm_set1_epi32(16);
    for (int r = 0; r < n; r++)
    {
        tmp = _mm_srli_epi32(key, 15 - r);
        tmp = _mm_xor_si128(tmp, _mm_srli_epi32(data, 31));
        tmp = _mm_xor_si128(tmp, _mm_srli_epi32(data, 15));
        *(__m128i*)arr = _mm_xor_si128(_mm_and_si128(_mm_srli_epi32(data, 0), sse_mask_p0),
            _mm_xor_si128(
                _mm_xor_si128(_mm_and_si128(_mm_srli_epi32(data, 7), sse_mask_p1),
                    _mm_and_si128(_mm_srli_epi32(data, 17), sse_mask_p2)),
                _mm_xor_si128(_mm_and_si128(_mm_srli_epi32(data, 22), sse_mask_p3),
                    _mm_and_si128(_mm_srli_epi32(data, 26), sse_mask_p4))
            )
        );
        tmp = _mm_xor_si128(tmp, _mm_set_epi32(KEELOQ_NLF >> arr[3], KEELOQ_NLF >> arr[2], KEELOQ_NLF >> arr[1], KEELOQ_NLF >> arr[0]));
        data = _mm_xor_si128(_mm_and_si128(tmp, sse_mask_p0), _mm_slli_epi32(data, 1));
    }
    return data;
}



uint32_t KlOptPartDecrypt(uint32_t data, uint16_t key, uint8_t val)
{
    int n = 16;
    if ((data & 0x7) == 0)
    {
        data = data << 8 + val;
        n = 8;
        key <<= 8;
    }
    for (int r = 0; r < n; r++)
        data = (data << 1) ^ bit(data, 31) ^ bit(data, 15) ^ (uint32_t)bit(key, 15 - r) ^ bit(KEELOQ_NLF, g5(data, 0, 8, 19, 25, 30));
    return data;
}


uint16_t KlKeyPart(uint32_t data, uint16_t hint)
{
    uint16_t key = 0;
    for (int i = 0; i < 16; i++)
    {
        key = key ^ (bit(data, 0) ^ bit(data, 16) ^ bit(KEELOQ_NLF, g5(data, 1, 9, 20, 26, 31)) ^ bit(hint, i)) << i;
        data = (data >> 1) ^ (bit(hint, i) << 31);
    }
    return key;
}

__m128i sse_KlKeyPart(__m128i data, __m128i hint)
{
    __m128i tmp;
    __m128i key = _mm_set1_epi32(0);
    uint32_t arr[4];
    __m128i sse_mask_p0 = _mm_set1_epi32(1);
    __m128i sse_mask_p1 = _mm_set1_epi32(2);
    __m128i sse_mask_p2 = _mm_set1_epi32(4);
    __m128i sse_mask_p3 = _mm_set1_epi32(8);
    __m128i sse_mask_p4 = _mm_set1_epi32(16);
    for (int r = 0; r < 16; r++)
    {
        tmp = _mm_xor_si128(data, _mm_srli_epi32(data, 16));
        tmp = _mm_xor_si128(tmp, _mm_srli_epi32(hint, r));
        *(__m128i*)arr = _mm_xor_si128(_mm_and_si128(_mm_srli_epi32(data, 1), sse_mask_p0),
            _mm_xor_si128(
                _mm_xor_si128(_mm_and_si128(_mm_srli_epi32(data, 8), sse_mask_p1),
                    _mm_and_si128(_mm_srli_epi32(data, 18), sse_mask_p2)),
                _mm_xor_si128(_mm_and_si128(_mm_srli_epi32(data, 23), sse_mask_p3),
                    _mm_and_si128(_mm_srli_epi32(data, 27), sse_mask_p4))
            )
        );
        tmp = _mm_xor_si128(tmp, _mm_set_epi32(KEELOQ_NLF >> arr[3], KEELOQ_NLF >> arr[2], KEELOQ_NLF >> arr[1], KEELOQ_NLF >> arr[0]));
        tmp = _mm_slli_epi32(_mm_and_si128(tmp, sse_mask_p0), r);
        key = _mm_xor_si128(key, tmp);
        data = _mm_xor_si128(_mm_srli_epi32(data, 1), _mm_slli_epi32(_mm_srli_epi32(hint, r), 31));
    }
    return key;
}

bool KlLastKeyPart(uint32_t data1, uint16_t hint1, uint32_t data2, uint16_t hint2, uint16_t* key)
{
    uint16_t key1 = 0;
    uint16_t key2 = 0;
    for (int i = 0; i < 16; i++)
    {
        key1 = key1 ^ ((data1 ^ (data1 >> 16) ^ (KEELOQ_NLF >> g5e(data1)) ^ (hint1 >> i)) & 1) << i;
        key2 = key2 ^ ((data2 ^ (data2 >> 16) ^ (KEELOQ_NLF >> g5e(data2)) ^ (hint2 >> i)) & 1) << i;
        if (key1 != key2)
            return false;
        data1 = (data1 >> 1) ^ ((hint1 >> i) << 31);
        data2 = (data2 >> 1) ^ ((hint2 >> i) << 31);
    }
    *key = key1;
    return true;
}

void slide(int k)
{
    for (int a = 0x7788; a < 0x10000; a++)
    {
        uint16_t k0 = (uint16_t)a;
        for (int i = 0; i < DICT_SIZE; i+=4)
        {
            //plainE[i] = KlEncrypt(plain[i], k0, KEELOQ_PART);
            *(__m128i*)&plainE[i] = sse_KlEncrypt(*(__m128i*)&plain[i], _mm_set1_epi32(k0), KEELOQ_PART);
            //cipherD[i] = KlDecrypt(cipher[i], k0, KEELOQ_PART);
            *(__m128i*)&cipherD[i] = sse_KlDecrypt(*(__m128i*)&cipher[i], _mm_set1_epi32(k0), KEELOQ_PART);
        }
        for (int b = 0x100 / THREADS * k; b < 0x100 / THREADS * (k + 1); b++)
        //for (int b = 0xc400; b < 0xc500; b++)
        {
            uint16_t ps = (uint16_t)b;
            //memset(table, 0, DICT_SIZE * HASH_AMT * 2);
            for (int i = 0; i < DICT_SIZE; i++)
            {
                for (int j = 1; j <= table[i][0]; j++)
                {
                    table[i][j] = 0;
                }
                table[i][0] = 0;
            }
            for (int i = 0; i < DICT_SIZE; i+=4)
            {
                //k3_arr[i] = KlKeyPart((plain[i] << 16) ^ ps, (uint16_t)(plain[i] >> 16));
                *(__m128i*)& k3_arr[i] = sse_KlKeyPart(_mm_xor_si128(_mm_slli_epi32(*(__m128i*) & plain[i], 16), _mm_set1_epi32(ps)), _mm_srli_epi32(*(__m128i*) & plain[i], 16));
                //cipherDD[i] = KlDecrypt(cipherD[i], k3_arr[i], KEELOQ_PART);
                *(__m128i*)& cipherDD[i] = sse_KlDecrypt(*(__m128i*) & cipherD[i], *(__m128i*) & k3_arr[i], KEELOQ_PART);
            }
            for (int i = 0; i < DICT_SIZE; i++)
            {
                uint16_t lo = cipherDD[i] & 0xFFFF;
                table[lo][(table[lo][0] % (TABLE_SIZE - 1)) + 1] = i;
                table[lo][0]++;
                if (table[lo][0] >= TABLE_SIZE)
                {
                    cout << "Mem" << endl;
                }
            }
            for (int i = 0; i < DICT_SIZE; i+=4)
            {
                //k1_arr[i] = KlKeyPart(plainE[i], ps);
                *(__m128i*)& k1_arr[i] = sse_KlKeyPart(*(__m128i*) & plainE[i], _mm_set1_epi32(ps));
                //cipherE[i] = KlEncrypt(cipher[i], k1_arr[i], KEELOQ_PART);
                *(__m128i*)& cipherE[i] = sse_KlEncrypt(*(__m128i*) & cipher[i], *(__m128i*) & k1_arr[i], KEELOQ_PART);
            }
            for (int i = 0; i < DICT_SIZE; i++)
            {
                for (int j = 1; j <= table[cipherE[i] >> 16][0]; j++)
                {
                    uint16_t lol = table[cipherE[i] >> 16][j % TABLE_SIZE];
                    uint16_t k2;
                    if (KlLastKeyPart((plainE[i] >> 16) ^ (ps << 16), (uint16_t)plain[lol], cipherE[i], (uint16_t)(cipherDD[lol] >> 16), &k2))
                    {
                        uint64_t key = (((uint64_t)(k3_arr[lol])) << 48) + ((uint64_t)k2 << 32) + ((uint64_t)k1_arr[i] << 16) + k0;
                        if (cipher[0] == KlEncrypt(plain[0], key, KEELOQ_FULL))
                        {
                            cout << format("Success! Key: {:x}", key) << endl;
                            flag = false;
                        }
                    }
                }
            }
        }
        break;
    }
}

int main(void)
{
    ifstream inFile("./test_data_set.txt", fstream::in);
    for (int i = 0; i < DICT_SIZE; i++)
        inFile >> hex >> plain[i] >> cipher[i];
    cout << "File readed" << endl;
    //for (int i = 0; i < DICT_SIZE; i++)
    //    cout << hex << plain[i] << " " << cipher[i] << endl;

    clock_t startTime = clock();

    thread threads[THREADS];
    for (int i = 0; i < THREADS; i++)
        threads[i] = thread(slide, i);
    for (int i = 0; i < THREADS; i++)
        threads[i].join();

    clock_t endTime = clock();
    cout << (float)(endTime - startTime) / CLOCKS_PER_SEC << endl;

    cin.get();

    return 0;
}