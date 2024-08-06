#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling AI essentials Module1 -- dpex Pair Wise - 5 of 5 pair_wise_numpy.py
python -Wignore lab/pair_wise_numpy.py