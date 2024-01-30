#include <iostream>
#include <fstream>
#include <thread>
#include <ctime>
#include <format>

#define THREADS 16

#define KEELOQ_NLF              0x3A5C742E
#define bit(x, n)               (((x) >> (n))&1)
#define g5(x, a, b, c, d, e)    (bit(x, a) + bit(x, b) * 2 + bit(x, c) * 4 + bit(x, d) * 8 + bit(x, e) * 16)

#define DICT_SIZE 0x10000

#define HASH_AMT 15

uint32_t plain[DICT_SIZE];
uint32_t cipher[DICT_SIZE];
thread_local uint32_t plainE[DICT_SIZE];
thread_local uint32_t plainD[DICT_SIZE];
thread_local uint32_t cipherD[DICT_SIZE];
thread_local uint32_t cipherDD[DICT_SIZE];
thread_local uint16_t k3_arr[DICT_SIZE];
thread_local uint16_t h_arr[DICT_SIZE][HASH_AMT];

bool flag = true;

using namespace std;

uint32_t KlFullEncrypt(uint32_t data, uint64_t key)
{
    for (int r = 0; r < 528; r++)
        data = (data >> 1) ^ ((bit(data, 0) ^ bit(data, 16) ^ (uint32_t)bit(key, r & 63) ^ bit(KEELOQ_NLF, g5(data, 1, 9, 20, 26, 31))) << 31);
    return data;
}

uint32_t KlFullDecrypt(uint32_t data, uint64_t key)
{
    for (int r = 0; r < 528; r++)
        data = (data << 1) ^ bit(data, 31) ^ bit(data, 15) ^ (uint32_t)bit(key, (15 - r) & 63) ^ bit(KEELOQ_NLF, g5(data, 0, 8, 19, 25, 30));
    return data;
}

uint32_t KlPartEncrypt(uint32_t data, uint16_t key)
{
    for (int r = 0; r < 16; r++)
        data = (data >> 1) ^ ((bit(data, 0) ^ bit(data, 16) ^ (uint32_t)bit(key, r) ^ bit(KEELOQ_NLF, g5(data, 1, 9, 20, 26, 31))) << 31);
    return data;
}

uint32_t KlPartDecrypt(uint32_t data, uint16_t key)
{
    for (int r = 0; r < 16; r++)
        data = (data << 1) ^ bit(data, 31) ^ bit(data, 15) ^ (uint32_t)bit(key, 15 - r) ^ bit(KEELOQ_NLF, g5(data, 0, 8, 19, 25, 30));
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

bool KlLastKeyPart(uint32_t data, uint16_t hint, uint32_t data2, uint16_t hint2, uint16_t* x)
{
    uint16_t key = 0;
    uint16_t key2 = 0;
    for (int i = 0; i < 16; i++)
    {
        key = key ^ (bit(data, 0) ^ bit(data, 16) ^ bit(KEELOQ_NLF, g5(data, 1, 9, 20, 26, 31)) ^ bit(hint, i)) << i;
        key2 = key2 ^ (bit(data2, 0) ^ bit(data2, 16) ^ bit(KEELOQ_NLF, g5(data2, 1, 9, 20, 26, 31)) ^ bit(hint2, i)) << i;
        if (key != key2)
            return false;
        data = (data >> 1) ^ (bit(hint, i) << 31);
        data2 = (data2 >> 1) ^ (bit(hint2, i) << 31);
    }
    *x = key;
    return true;
}

void slide(int k)
{
    for (int a = 0x7788; a < 0x10000; a++)
    {
        uint16_t k0 = (uint16_t)a;
        for (int i = 0; i < DICT_SIZE; i++)
        {
            plainE[i] = KlPartEncrypt(plain[i], k0);
            cipherD[i] = KlPartDecrypt(cipher[i], k0);
        }
        for (int b = 0x1000 / THREADS * k; b < 0x1000 / THREADS * (k + 1); b++)// 3.4 sec, 16 threads
        //for (int b = 0xc400; b < 0xc500; b++)
        {
            uint16_t ps = (uint16_t)b;
            //memset(h_arr, 0, DICT_SIZE * HASH_AMT * 2);
            for (int i = 0; i < DICT_SIZE; i++)
            {
                for (int j = 0; j < HASH_AMT; j++) {
                    if (h_arr[i][j])
                        h_arr[i][j] = 0;
                    else
                        break;
                }
            }
            for (int i = 0; i < DICT_SIZE; i++)
            {
                k3_arr[i] = KlKeyPart((plain[i] << 16) ^ ps, (uint16_t)(plain[i] >> 16));
                cipherDD[i] = KlPartDecrypt(cipherD[i], k3_arr[i]);
                uint16_t lo = cipherDD[i] & 0xFFFF;
                for (int j = 0; j < HASH_AMT; j++) {
                    if (h_arr[lo][j] == 0) {
                        h_arr[lo][j] = i;
                        break;
                    }
                    if (j == HASH_AMT - 1)
                    {
                        cout << "Mem" << endl;
                    }
                }
            }
            for (int i = 0; i < DICT_SIZE; i++)
            {
                uint16_t k1 = KlKeyPart(plainE[i], ps);
                uint32_t cipherE = KlPartEncrypt(cipher[i], k1);
                for (int j = 0; j < HASH_AMT; j++)
                {
                    if (h_arr[cipherE >> 16][j] != 0)
                    {
                        uint16_t lol = h_arr[cipherE >> 16][j];
                        uint16_t k2;
                        if (KlLastKeyPart((plainE[i] >> 16) ^ (ps << 16), (uint16_t)plain[lol], cipherE, (uint16_t)(cipherDD[lol] >> 16), &k2))
                        {
                            uint64_t key = (((uint64_t)(k3_arr[lol])) << 48) + ((uint64_t)k2 << 32) + ((uint64_t)k1 << 16) + k0;
                            if (cipher[0] == KlFullEncrypt(plain[0], key))
                            {
                                cout << format("Success! Key: {:x}", key) << endl;
                                flag = false;
                            }
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
    clock_t startTime = clock();

    ifstream inFile("./test_data_set.txt", fstream::in);
    for (int i = 0; i < DICT_SIZE; i++)
        inFile >> hex >> plain[i] >> cipher[i];
    cout << "File readed" << endl;
    //for (int i = 0; i < DICT_SIZE; i++)
    //    cout << hex << plain[i] << " " << cipher[i] << endl;

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