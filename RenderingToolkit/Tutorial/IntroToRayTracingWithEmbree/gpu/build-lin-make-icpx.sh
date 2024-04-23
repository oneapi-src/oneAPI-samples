#!/bin/bash

if [ ! -z ${ONEAPI_ROOT+x} ]; then
  source "${ONEAPI_ROOT}/setvars.sh"
else
  source /opt/intel/oneapi/setvars.sh
fi

CXX_COMPILER=icpx

BUILD_COMMAND="cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_INSTALL_PREFIX=.. .."
echo "Build Command: ${BUILD_COMMAND}" > build-command.txt
echo "CXX_COMPILER ${CXX_COMPILER}" >> build-command.txt
echo "CXX_COMPILER VERSION: $(${CXX_COMPILER} --version)" >> build-command.txt
rm -rf build
mkdir build
pushd build 
eval ${BUILD_COMMAND}
cmake --build . 
cmake --install .
popd
cp build-command.txt bin/build-command.txt

