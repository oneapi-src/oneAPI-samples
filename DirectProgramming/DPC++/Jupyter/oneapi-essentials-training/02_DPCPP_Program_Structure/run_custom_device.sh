#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module2 -- DPCPP Program Structure sample - 5 of 6 custom_device_sample.cpp
dpcpp lab/custom_device_sample.cpp -o bin/custom_device_sample
bin/custom_device_sample
echo "########## Done with the run"
