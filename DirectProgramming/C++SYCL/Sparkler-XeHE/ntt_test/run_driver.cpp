#include "include/driver.h"

void run_driver(int n, int time_loop, int rns_base_sz, size_t queue_num)
{

    {
    std::cout << "---------------benchmarking uint64_t---------------" << std::endl;
    int prime_length = sizeof(uint64_t)*8 - 4;
    std::vector<int> mods(rns_base_sz, prime_length);
    Driver<uint64_t> driver_64;
    driver_64.benchmark_interface(n, time_loop, prime_length, &mods, queue_num);
    std::cout << "---------------  finished uint64_t  ---------------" << std::endl;
    std::cout << std::endl;
    }
    {
    std::cout << "---------------benchmarking uint32_t---------------" << std::endl;
    Driver<uint32_t> driver_32;
    int prime_length = sizeof(uint32_t)*8 - 4; 
    rns_base_sz = int(ceil(((float(sizeof(uint64_t)*8 - 4)*rns_base_sz) / prime_length)));

    std::vector<int> mods(rns_base_sz, prime_length);   
    driver_32.benchmark_interface(n, time_loop, prime_length, &mods, queue_num);
    std::cout << "---------------  finished uint32_t  ---------------" << std::endl;
    std::cout << std::endl;
    }
}