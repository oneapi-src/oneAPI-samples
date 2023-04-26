#!/bin/bash

if [ ! -z ${ONEAPI_ROOT+x} ]; then
  source "${ONEAPI_ROOT}/setvars.sh"
else
  source /opt/intel/oneapi/setvars.sh
fi

CXX_COMPILER="${ONEAPI_ROOT}/compiler/latest/linux/bin-llvm/clang++"

BUILD_COMMAND="cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER=${CXX_COMPILER} -D CMAKE_INSTALL_PREFIX=.. .."
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

