#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module10 -- DPCPP Graphs and Dependencies sample - 4 of 10 accessors_WAR_WAW.cpp
dpcpp lab/accessors_WAR_WAW.cpp -o bin/accessors_WAR_WAW
if [ $? -eq 0 ]; then bin/accessors_WAR_WAW; fi

