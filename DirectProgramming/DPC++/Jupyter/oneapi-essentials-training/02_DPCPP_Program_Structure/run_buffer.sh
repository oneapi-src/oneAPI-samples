#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module2 -- DPCPP Program Structure sample - 2 of 6 buffer_sample.cpp
dpcpp lab/buffer_sample.cpp -o bin/buffer_sample
bin/buffer_sample
echo "########## Done with the run"
