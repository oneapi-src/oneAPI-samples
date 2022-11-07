#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module7 -- oneDPL Extension APIs - 8 of 12 transform_iterator.cpp
dpcpp lab/transform_iterator.cpp -w -o bin/transform_iterator
bin/transform_iterator
