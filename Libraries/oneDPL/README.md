# Intel® oneAPI DPC++ Library (oneDPL) Code Samples

| Code Sample Name                                                                                   | Supported Intel® Architecture(s) | Description
|:---                                                                                                |:---                              |:---
| [dynamic_selection](https://github.com/uxlfoundation/oneDPL/tree/main/examples/dynamic_selection)  | GPU, CPU                         | The nstream and sepia-filter-ds samples demonstrate the dynamic selection feature in oneDPL on five different policies - fixed CPU, fixed GPU, round robin, dyanmic load, and auto-tune.
| [maxloc_reductions](https://github.com/uxlfoundation/oneDPL/tree/main/examples/maxloc_reductions)  | GPU, CPU                         | Code sample shows four ways of finding the location of the maximum value in an array - using the SYCL reduction operator, using oneDPL max_element and distance functions, using oneDPL and SYCL buffers, and using oneDPL and USM.
| [pSTL_offload](https://github.com/uxlfoundation/oneDPL/tree/main/examples/pSTL_offload)            | GPU, CPU                         | Shows how to offload standard C++ pSTL code to the CPU and GPU in oneDPL by using the -fsycl-pstl-offload compiler option.
