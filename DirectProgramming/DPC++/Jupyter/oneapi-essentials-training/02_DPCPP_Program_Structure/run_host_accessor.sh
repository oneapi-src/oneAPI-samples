#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module2 -- DPCPP Program Structure sample - 3 of 6 host_accessor_sample.cpp
dpcpp lab/host_accessor_sample.cpp -o bin/host_accessor_sample
bin/host_accessor_sample
echo "########## Done with the run"
