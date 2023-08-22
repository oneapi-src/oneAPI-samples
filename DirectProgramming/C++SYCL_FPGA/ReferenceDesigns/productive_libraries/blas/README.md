# `BLAS`

This directory contains FPGA reference designs for the standard BLAS kernels defined in [oneMKL](https://oneapi-src.github.io/oneMKL/domains/blas/blas.html). The row-major, USM-based SYCL interface is supported.

Kernels of similar computes are grouped and generalized into a single systolic array so that the array can be dynamically reconfigured to simulate all the kernels, minimizing maintenance cost without losing performance. Below are the kernels supported in this release:

## `Level 1 kernels`

A [dot-product systolic array](reconfigurable_dotprod/README.md) supports

| Kernel                                                                 | Formula                                               | Description                                                                                        | VARIATION                    |
| ---------------------------------------------------------------------- | ----------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ---------------------------- |
| [dot](https://oneapi-src.github.io/oneMKL/domains/blas/dot.html)       | $\vec{X}\cdot \vec{Y}$                                | Dot product.                                                                                       | sdot, ddot, dsdot            |
| [sdsdot](https://oneapi-src.github.io/oneMKL/domains/blas/sdsdot.html) | $sb+\vec{X}\cdot \vec{Y}$                             | Return a single-precision result with a dot product of two vectors accumulated in double-precision | sdsdot                       |
| [dotc](https://oneapi-src.github.io/oneMKL/domains/blas/dotc.html)     | $\overline{\vec{X}}\cdot \vec{Y}$                     | A dot product between two complex vectors, conjugating the first of them                           | cdotc, zdotc                 |
| [dotu](https://oneapi-src.github.io/oneMKL/domains/blas/dotu.html)     | $\vec{X}\cdot \vec{Y}$                                | A dot product between two complex vectors                                                          | cdotu, zdotu                 |
| [nrm2](https://oneapi-src.github.io/oneMKL/domains/blas/nrm2.html)     | $\parallel \vec{X} \parallel$                         | Euclidean norm of a vector                                                                         | snrm2, dnrm2, scnrm2, dznrm2 |
| [asum](https://oneapi-src.github.io/oneMKL/domains/blas/asum.html)     | sum of $\mid Re(x_i)\mid+\mid Im(x_i)\mid, \forall i$ | Sum of the magnitudes of elements                                                                  | sasum, dasum, scasum, dzasum |

The `VARIATION` column shows the variations of each kernel, usually the kernel name prefixed by the output/input data types. A data type can be `s` (single-precision), `d`(double-precision), `c`(complex single-precision) or `z`(complex double-precision).

A [vector-addition systolic array](reconfigurable_vecadd/README.md) supports
| Kernel            | Formula                                           | Description                                                                                                                                |VARIATION |  Note |
| ----------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |-----| --- |
| [axpy](https://oneapi-src.github.io/oneMKL/domains/blas/axpy.html)   | $\alpha * \vec{X}+\vec{Y}$                           | Vector addition                                                 | saxpy, daxpy, caxpy, zaxpy ||
| [scal](https://oneapi-src.github.io/oneMKL/domains/blas/scal.html)   | $\alpha * \vec{X}$                                   | Scale a vector                                                  | sscal, dscal, cscal, zscal | csscal and zdscal are to be supported in the next release |
| [copy](https://oneapi-src.github.io/oneMKL/domains/blas/copy.html)   | $\vec{Y}\leftarrow\vec{X}$                        | Copy a vector                                                      | scopy, dcopy, ccopy, zcopy | |

## `Level 3 kernels`

A [matrix-multiply systolic array](reconfigurable_matmul/README.md) supports
 Kernel          | Formula             | Description       |VARIATION | Note |
| --------------- | ------------------- | ----------|-----|---|
| [gemm](https://oneapi-src.github.io/oneMKL/domains/blas/gemm.html) | $\alpha * op(A) * op(B)+\beta * C$ |Multiplication of general matrices. $op(X)$ is one of $X$, $X^T$, and $X^H$ | sgemm, dgemm, cgemm, zgemm|Half and bfloat16 are to be supported in future|
| [symm](https://oneapi-src.github.io/oneMKL/domains/blas/symm.html) | $\alpha * A* B+\beta * C$, or  $\alpha * B * A+\beta * C$ | $A$ is a symmetric matrix | ssymm, dsymm, csymm, zsymm ||
| [hemm](https://oneapi-src.github.io/oneMKL/domains/blas/hemm.html) |$\alpha * A * B+\beta * C$, or  $\alpha * B * A+\beta * C$ | $A$ is a Hermitian matrix | chemm, zhemm ||
| syrk | $C \leftarrow \alpha * op(A) * op(A)^T + \beta * C$ |$op(X)=X$ or $op(X) = X^T$, $C$ is a symmtric matrix. | ssyrk, dsyrk, csyrk, zsyrk| To be available in the next release |
| herk | $C \leftarrow \alpha * op(A) * op(A)^H + \beta * C$ |$op(X)=X$ or $op(X) = X^H$, $C$ is a Hermitian matrix. |cherk, zherk| To be available in the next release |

### Restrictions

* Matrix storage: row-major.
* Data types: `s`, `d`, `c`, `z`.
* Data sizes: For memory efficiency, input and output data must be loaded and stored as a series of short vectors from/to the device memory. Therefore, the dimensions of the data must be multiples of the length of a short vector. This restriction is to be removed in the next release.

## `File structure`

All the kernels are put under the `blas` directory. Every kernel has the following files under it:

* `api.hpp` - The API to invoke the kernel in any SYCL application.
* `test.cpp` - Unit tests for correctness from [oneMKL's test suite](https://github.com/oneapi-src/oneMKL/blob/develop/tests/unit_tests/blas/), with slight changes to respect the above restrictions.
* `demo.cpp` - Demonstrating how to invoke the kernel on a real FPGA hardware.
* `CMakeLists.txt` - A cmake script to build the kernel.

The reconfigurable systolic arrays (named as `reconfigurable-*`) are also under the `blas` directory. Every array has the following files under it:

* `api.hpp` - The API to invoke the array.
* `spec.cpp`: A specification of the array in a productive language, namely [T2SP](#user-content-reference). From this specification, SYCL files will be generated by a pre-installed T2SP compiler. The SYCL files are then synthesized into a bitstream for an FPGA hardware.
* `parameters.h` : Sizes of the array. There is a `tiny` and a `large` configuration for testing correctness and performance, respectively.
* `CMakeLists.txt` - A cmake script to build the array.
* `README.md` - A short description of the array.

## Environment requirement

We assume your machine has OneAPI enabled for A10 or S10 FPGA. For example, on DevCloud,

    ```shell
    # Ask for a compute node
    login-2:~$ devcloud_login
                    Choose either option 2) Arria 10 - OneAPI, ...
                               or option 4) Stratix 10 - OneAPI, ...
                               
    # On the compute node
    source /glob/development-tools/versions/oneapi/2023.2.0.1/oneapi/setvars.sh --force
    ```

## Batch tests

To batch build and run the tests and demos of all the kernels:

    ```shell
    PATH_TO_ONEAPI_SAMPLES/DirectProgramming/C++SYCL_FPGA/ReferenceDesigns/productive_libraries/tools/batch.sh a10|s10
    ```

## Test an individual kernel

1. Configure the build system.

    ```shell
    # Replace KERNEL with a specific kernel to build, e.g. gemm
    cd PATH_TO_ONEAPI_SAMPLES/DirectProgramming/C++SYCL_FPGA/ReferenceDesigns/productive_libraries/blas/KERNEL 
    
    mkdir -p build
    cd build
    ```

    Then for Intel Arria® 10 GX FPGA:

    ```shell
    cmake ..
    ```

    For Intel Stratix® 10 SX:

    ```shell
    cmake .. -DFPGA_DEVICE=intel_s10sx_pac:pac_s10
    ```

2. Test correctness.

    ```shell
    # Build tiny systolic arrays and run them on an FPGA emulator
    make tests
    ../bin/tests.sh
    ```

3. Build a demo applictaion and test performance

    Replace `VARIATION` below with a specific variation of the kernel as listed in the tables above, and replace `HW` below with either `a10` or `s10`.

    ```shell
    # Optional: if to avoid synthesis of the kernel's underlying systolic array, pre-generated bitstream (as well as SYCL file and reports) for the array can be installed.
    make install_VARIATION_large_HW

    # Synthesize the systolic array and link it with a demo application of the kernel
    make demo_VARIATION_large_HW
    ```

    Then for A10:

    ```shell
    # Unsign the bitstream. Otherwise, there is an "Error writing bitstream to FPGA" due to the security feature of devstack 1.2.1
    make unsign_VARIATION_large_a10

    # Demo on the hardware
    ../bin/demo_VARIATION_large_a10.unsigned
    ```

    For S10:

    ```shell
    ../bin/demo_VARIATION_large_s10
    ```

    Take `sgemm` with a large systolic array on A10 for example:

    ```shell
    # Optional: install the pre-generated bitstream.
    make install_sgemm_large_a10

    # Generate a demo application on the FPGA hardware
    make demo_sgemm_large_a10

    # A10 specific: Unsign the bitstream
    make unsign_sgemm_large_a10

    # Demo on the hardware
    ../bin/demo_sgemm_large_a10.unsigned
    ```

4. Delete all generated files for a kernel

    ```shell
    make clean_VARIATION_(tiny|large)_HW
    ```

    For example:

    ```shell
    make clean_sgemm_large_a10
    ```

## Known issues

* Level 1 kernels suffer from [an issue](https://github.com/haoxiaochen/t2sp/issues/40) that two input vectors cannot be allocated two different DDR channels exclusively in SYCL in USM memory model.

* dsdot and sdsdot further suffer from [another issue](https://github.com/haoxiaochen/t2sp/issues/39) that T2SP compiler converts floats to doubles too early in the datapaths

* Synthesis of level 3 kernels with complex types [fail](https://github.com/haoxiaochen/t2sp/issues/33) or [overtime](https://github.com/haoxiaochen/t2sp/issues/34) in some configurations.

* The systolic array of matrix multiply [scales worse](https://github.com/haoxiaochen/t2sp/issues/41) in SYCL than in OpenCL. Especially, DSP utilization on S10 is low, and parameter tuning is ongoing.

* Running a demo occasionally shows segmentation fault during SYCL queue destruction. Usually, re-run it would be fine.

## Next release

* Further tuned performance
    * Near-peak performance for level 1 kernels and complex-typed kernels after addressing the above known issues
    * Increased DSP utilization of level 3 kernels on S10

* Automatic parameter tuning
    * Linear programming models for the systolic arrays

* More kernels
    * level 1: csscal, zdscal
    * levle 2: gemv, gbmv, hemv, hbmv, hpmv, symv, sbmv, spmv
    * level 3: syrk, herk, syr2k, her2k

* Further improved readability of generated SYCL files

* Remove the restriction on data sizes

## Reference

[T2SP](https://github.com/IntelLabs/t2sp) (Temporal To Spatial Programming, previously called T2S) language constructs systolic arrays for dense tensor computes. For convenience, a latest experimental version of the T2SP compiler, namely Lasa, has been pre-installed in this website, and is automatically invoked when making tests or demos.

The key point of the language is to express a systolic array as two dataflow graphs, one for compute, the other for data movement. To understand more details, please read

1. Lasa: Abstraction and Specialization for Productive and Performant Linear Algebra on FPGAs. Xiaochen Hao, Mingzhe Zhang, Ce Sun, Zhuofu Tao, Hongbo Rong, Yu Zhang, Lei He, Eric Petit, Wenguang Chen, Yun Liang. FCCM, 2023. https://ieeexplore.ieee.org/document/10171577.
