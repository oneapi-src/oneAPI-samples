#include "include/driver.h"

void show_usage(){
    std::cout << "./ntt_test \t\t\t\t\t\t| run on default parameters" << std::endl;
    std::cout << "./ntt_test [queue_num] \t\t\t\t\t| run on specified # of queues" << std::endl;
    std::cout << "./ntt_test [rns_size] [queue_num] \t\t\t| run on specified RNS size and # of queues" << std::endl;
    std::cout << "./ntt_test [poly_size] [rns_size] [queue_num] \t\t| run on specified poly size, RNS size and # of queues" << std::endl;
    std::cout << "./ntt_test [poly_size] [rns_size] [queue_num] [loops] \t| run on specified poly size, RNS size, # of queues and loop iterations" << std::endl;
}

int main(int argc, char** argv){
    if (argc == 1) run_driver(1024*4, 100, 16);
    else if (argc == 2) {
        std::string arg = argv[1];
        if (arg=="-h" || arg=="--help") show_usage();
        else run_driver(1024*4, 100, 16, stoi(argv[1]));
    }
    else if (argc == 3) run_driver(1024*4, 100, stoi(argv[1]), stoi(argv[2]));
    else if (argc == 4) run_driver(1024*(stoi(argv[1])/2), 100, stoi(argv[2]), stoi(argv[3]));
    else if (argc == 5) run_driver(1024*(stoi(argv[1])/2), stoi(argv[4]), stoi(argv[2]), stoi(argv[3]));
    // std::cout << "In main" << std::endl;
    return 0;
}
