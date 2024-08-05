#!/bin/bash

rm -rf build
build="$PWD/build"
[ ! -d "$build" ] && mkdir -p "$build"
cd build &&
cmake .. &&
make run_gpu_linear
