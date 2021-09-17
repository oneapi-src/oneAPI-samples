#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module2 -- DPCPP Program Structure sample - 4 of 6 buffer_destruction2.cpp
dpcpp lab/buffer_destruction2.cpp -o bin/buffer_destruction2
if [ $? -eq 0 ]; then bin/buffer_destruction2; fi

