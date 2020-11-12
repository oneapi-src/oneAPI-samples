#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module4 -- DPCPP Sub Groups - 2 of 3 sub_group_shuffle.cpp
dpcpp lab/sub_group_shuffle.cpp 
if [ $? -eq 0 ]; then ./a.out; fi

