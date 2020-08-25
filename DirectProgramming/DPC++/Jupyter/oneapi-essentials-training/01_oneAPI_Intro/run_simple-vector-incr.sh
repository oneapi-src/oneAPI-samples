#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module1 -- oneAPI Intro sample - 2 of 2 simple-vector-incr.cpp
dpcpp lab/simple-vector-incr.cpp -o bin/simple-vector-incr
bin/simple-vector-incr
