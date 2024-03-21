#!/bin/bash
source /opt/intel/oneapi/setvars.sh --force

export rkcommon_DIR=/opt/intel/oneapi/rkcommon/latest/lib/cmake/rkcommon

/bin/echo "##" $(whoami) "is compiling OSPRay_Intro with denoise (ospTutorial_denoise)"
[ ! -d build_denoise ] && mkdir build_denoise
cd build_denoise
rm -rf *
cmake ../script_denoise
make -j12 $1
