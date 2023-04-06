/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2020, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#include "perf_app.hpp"
/*
* Am code dump
* Add
* AddMod
* Mul
* MulMod
* Mulx2
* Mulx2Mod
*/
int main(int argc, char* argv[]){
    xehe::dpcpp::Context ctx;
    std::cout << "Hello with DPC++! Context created: " << &ctx << std::endl;

    DumpUtil<uint32_t>(64, 16, 28);
    DumpUtil<uint64_t>(64, 16, 60);
}