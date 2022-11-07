#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module7 -- oneDPL Extension APIs - 4 of 12 binary_search.cpp
dpcpp lab/binary_search.cpp -w -o bin/binary_search
bin/binary_search
