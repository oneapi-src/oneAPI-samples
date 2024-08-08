#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1

export ONEAPI_DEVICE_SELECTOR=level_zero:gpu

if [ $? -eq 0 ]; then ./offload_mm.out; fi
