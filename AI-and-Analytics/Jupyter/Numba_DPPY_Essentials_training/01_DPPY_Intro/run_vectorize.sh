#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling AI Essentials Module1 -- DPPY Intro sample - 1 of 2 vectorize.py
python lab/vectorize.py

