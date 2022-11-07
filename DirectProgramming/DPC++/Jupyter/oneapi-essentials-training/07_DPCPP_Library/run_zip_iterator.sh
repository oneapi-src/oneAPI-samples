#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module7 -- oneDPL Extension APIs - 7 of 12 zip_iterator.cpp
dpcpp lab/zip_iterator.cpp -w -o bin/zip_iterator
bin/zip_iterator
