#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module10 -- DPCPP Graphs and Dependencies sample - 3 of 10 accessors_RAW.cpp
dpcpp lab/accessors_RAW.cpp -o bin/accessors_RAW
if [ $? -eq 0 ]; then bin/accessors_RAW; fi

