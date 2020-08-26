#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module2 -- DPCPP Program Structure sample - 1 of 6 gpu_sample.cpp
dpcpp lab/gpu_sample.cpp -o bin/gpu_sample
bin/gpu_sample
echo "########## Done with the run"
