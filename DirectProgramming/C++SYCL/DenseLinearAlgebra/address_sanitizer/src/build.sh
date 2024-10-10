#!/bin/sh

export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
# export ONEAPI_DEVICE_SELECTOR=opencl:cpu

##
## global memory
##

echo
echo === global memory ===

icpx -fsycl -Xarch_device -fsanitize=address -g -O2 -o nd_range_reduction nd_range_reduction.cpp
./nd_range_reduction

##
## group local memory
##

echo
echo === group local memory ===

icpx -fsycl -Xarch_device -fsanitize=address -g -O2 -o group_local group_local.cpp
./group_local

##
## local accessor
##

echo
echo === local accessor ===

icpx -fsycl -Xarch_device -fsanitize=address -g -O2 -o matmul_broadcast matmul_broadcast.cpp
./matmul_broadcast

##
## sycl::buffer
##

echo
echo === sycl::buffer ===

icpx -fsycl -Xarch_device -fsanitize=address -g -O2 -o local_stencil local_stencil.cpp
./local_stencil

##
## device global
##

echo
echo === device global ===

icpx -fsycl -Xarch_device -fsanitize=address -g -O2 -o device_global device_global.cpp
./device_global

##
## use-after-free
##

echo
echo === use-after-free ===

# To get more accurate allocate/release stack information, you need to use "-O0"
# Needs to enable "quarantine cache"
icpx -fsycl -Xarch_device -fsanitize=address -g -O0 -o map map.cpp
UR_LAYER_ASAN_OPTIONS=quarantine_size_mb:1 ./map

##
## misaligned access
##

echo
echo === misaligned access ===

icpx -fsycl -Xarch_device -fsanitize=address -g -O2 -o misalign-long misalign-long.cpp
./misalign-long

##
## double-free
##

echo
echo === double-free ===

# Need to enable "quarantine cache"
icpx -fsycl -Xarch_device -fsanitize=address -g -O0 -o array_reduction array_reduction.cpp
UR_LAYER_ASAN_OPTIONS=quarantine_size_mb:1 ./array_reduction

##
## bad free
##

echo
echo === bad free ===

icpx -fsycl -Xarch_device -fsanitize=address -g -O0 -o bad_free bad_free.cpp
UR_ENABLE_LAYERS=UR_LAYER_ASAN ./bad_free
