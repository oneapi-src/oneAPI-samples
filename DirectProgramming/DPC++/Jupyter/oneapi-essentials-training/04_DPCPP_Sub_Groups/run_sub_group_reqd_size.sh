#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module4 -- DPCPP Sub Groups - 2 of 6 sub_group_reqd_size.cpp
dpcpp lab/sub_group_reqd_size.cpp 
if [ $? -eq 0 ]; then ./a.out; fi


