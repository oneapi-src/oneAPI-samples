#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) "is compiling Welcome Module-- 1 of 1 hello.cpp"
icpx -fsycl src/hello.cpp -o src/hello
if [ $? -eq 0 ]; then src/hello; fi

