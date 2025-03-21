#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
echo "export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE"
echo
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
sycl-ls
echo
icpx -fsycl lab/vectoradd_single_device.cpp -w
if [ $? -eq 0 ]; then ./a.out; fi
