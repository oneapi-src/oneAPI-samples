#!/bin/bash
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1
/bin/echo "##" $(whoami) "is running OSPRay_Intro with denoise (ospTutorial_denoise)"
bin/ospTutorialDenoise
