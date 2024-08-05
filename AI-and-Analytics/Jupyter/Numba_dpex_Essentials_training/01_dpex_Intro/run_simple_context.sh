#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling AI essentials Module1 -- dpex Intro sample - 1 of 2 simple_context.py
python -Wignore lab/simple_context.py
