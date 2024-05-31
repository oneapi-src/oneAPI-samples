#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1

export ONEAPI_DEVICE_SELECTOR="level_zero:0;level_zero:1;level_zero:2"

if [ $? -eq 0 ]; then ./offload_vadd.out; fi
