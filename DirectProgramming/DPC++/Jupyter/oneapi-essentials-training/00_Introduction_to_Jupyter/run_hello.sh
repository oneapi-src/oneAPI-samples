#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) "is compiling Welcome Module-- 1 of 1 hello.cpp"
dpcpp src/hello.cpp -o src/hello
src/hello
