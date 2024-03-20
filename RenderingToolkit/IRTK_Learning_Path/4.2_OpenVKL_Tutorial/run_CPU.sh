#!/bin/bash
source /opt/intel/oneapi/setvars.sh &>/dev/null

/bin/echo "##" $(whoami) is running vklTutorialCPU
bin/vklTutorialCPU
