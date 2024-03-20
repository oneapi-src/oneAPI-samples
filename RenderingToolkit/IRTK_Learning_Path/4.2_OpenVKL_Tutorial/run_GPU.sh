#!/bin/bash
source /opt/intel/oneapi/setvars.sh &>/dev/null

/bin/echo "##" $(whoami) is running vklTutorialGPU
bin/vklTutorialGPU
