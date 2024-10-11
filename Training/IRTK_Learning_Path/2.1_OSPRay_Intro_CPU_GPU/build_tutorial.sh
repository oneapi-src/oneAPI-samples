#!/bin/bash
source /opt/intel/oneapi/setvars.sh --force

export rkcommon_DIR=/opt/intel/oneapi/rkcommon/latest/lib/cmake/rkcommon

/bin/echo "##" $(whoami) "is compiling OSPRay_Intro"
[ ! -d build_tutorial ] && mkdir build_tutorial
cd build_tutorial
rm -rf *
cmake ../script_tutorial
make -j12 $1
