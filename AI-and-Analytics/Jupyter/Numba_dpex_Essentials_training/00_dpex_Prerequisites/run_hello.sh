#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling AI Essentials Module 0 -- DPPY Prerequisites - 2 of 5 hello_world.py
python src/hello_world.py