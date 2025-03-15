#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
echo "export ZE_FLAT_DEVICE_HIERARCHY=FLAT"
echo
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
sycl-ls
echo
icpx -fsycl lab/vectoradd_multi_device.cpp -w
if [ $? -eq 0 ]; then ./a.out; fi
