#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is debugging SYCL_Essentials Module10 -- GDB
gdb-oneapi -batch -command=lab/array-transform.gdb -q ./bin/array-transform

