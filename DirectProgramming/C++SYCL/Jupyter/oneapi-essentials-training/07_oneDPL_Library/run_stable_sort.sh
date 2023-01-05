#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module7 -- oneDPL Extension APIs - 12 of stable_sort_bykey.cpp
rm -rf stable_sort_by_key/build
cd stable_sort_by_key &&
mkdir build && cd build  # execute in this directory
cmake ..
cmake --build .  # or "make"
cmake --build . --target run  # or "make run"

