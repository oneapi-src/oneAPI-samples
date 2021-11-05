#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module7 -- oneDPL Extension APIs - 8 of 12 discard_iterator.cpp
dpcpp lab/discard_iterator.cpp -w -o bin/discard_iterator
bin/discard_iterator
