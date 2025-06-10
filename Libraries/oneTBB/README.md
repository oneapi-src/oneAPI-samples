# Intel® oneAPI Threading Building Blocks (oneTBB) Code Samples

| Code Sample Name                                                                                             | Supported Intel(r) Architecture(s) | Description
|:---                                                                                                          |:---                                |:---
| [tbb-async-sycl](https://github.com/uxlfoundation/oneTBB/tree/master/examples/sycl/tbb-async-sycl)           | GPU, CPU                           | The calculations are split between TBB Flow Graph asynchronous node that calls SYCL* kernel on GPU while TBB functional node does CPU part of calculations.
| [tbb-task-sycl](https://github.com/uxlfoundation/oneTBB/tree/master/examples/sycl/tbb-resumable-tasks-sycl)  | GPU, CPU                           | One TBB task executes SYCL code on GPU while another TBB task performs calculations using TBB parallel_for.
| [tbb-resumable-tasks-sycl](https://github.com/uxlfoundation/oneTBB/tree/master/examples/sycl/tbb-task-sycl)  | GPU, CPU                           | The calculations are split between TBB resumable task that calls SYCL kernel on GPU while TBB parallel_for does CPU part of calculations.
