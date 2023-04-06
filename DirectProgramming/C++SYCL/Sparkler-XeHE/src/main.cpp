/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2020, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#include "../include/util/sanity_check_gpu.hpp"

int main(int argc, char* argv[]){

    std::cout << "Hello from DPC++ and XeHE project " << std::endl;
    xehe::util::XeHE_sanity_check<uint32_t>(64);
    xehe::util::XeHE_sanity_check<uint64_t>(64);

}