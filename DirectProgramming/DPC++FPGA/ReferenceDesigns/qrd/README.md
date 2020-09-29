# QR Decomposition of Matrices
This DPC++ reference design demonstrates high-performance QR decomposition of complex matrices on FPGA.

***Documentation***: The [FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide)  provides comprehensive instructions for targeting FPGAs through DPC++. The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a resource for general target-independent DPC++ programming. 

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA; <br> Intel® PAC with Intel Stratix® 10 SX FPGA; <br> Intel Xeon® CPU E5-1650 v2 @ 3.50GHz (host machine)
| Software                          | Intel® oneAPI DPC++ Compiler (Beta) <br> Intel® FPGA Add-On for oneAPI Base Toolkit
| What you will learn               | Implementing a high performance FPGA version of the Gram-Schmidt QR decomposition algorithm.
| Time to complete                  | 1 hr (not including compile time)

_Notice: Limited support in Windows*; compiling for FPGA hardware is not supported in Windows*_


**Performance**
Please refer to performance disclaimer at the end of this README.

| Device                                         | Throughput
|:---                                            |:---
| Intel® PAC with Intel Arria® 10 GX FPGA        | 25k matrices/s for matrices of size 128 * 128
| Intel® PAC with Intel Stratix® 10 SX FPGA      | 7k matrices/s for matrices of size 256 * 256


## Purpose

This FPGA reference design demonstrates QR decomposition of matrices of complex numbers, a common operation employed in linear algebra. Matrix _A_ (input) is decomposed into a product of an orthogonal matrix _Q_ and an upper triangular matrix _R_.

The algorithms employed by the reference design are the Gram-Schmidt QR decomposition algorithm and the thin QR factorization method. Background information on these algorithms can be found in Wikipedia's [QR decomposition](https://en.wikipedia.org/wiki/QR_decomposition) article. The original algorithm has been modified and optimized for performance on FPGAs in this implementation.

QR decomposition is used extensively in signal processing applications such as beamforming, multiple-input multiple-output (MIMO) processing, and Space Time Adaptive Processing (STAP).


### Matrix dimensions and FPGA resources

The QR decomposition algorithm factors a complex _m_×_n_ matrix, where _m_ ≥ _n_. The algorithm computes the vector dot product of two columns of the matrix. In our FPGA implementation, the dot product is computed in a loop over the _m_ elements of the column. The loop is fully unrolled to maximize throughput. As a result, *m* complex multiplication operations are performed in parallel on the FPGA, followed by sequential additions to compute the dot product result. 

We use the compiler flag `-fp-relaxed`, which permits the compiler to reorder floating point additions (i.e. to assume that floating point addition is commutative). The compiler uses this freedom to reorder the additions so that the dot product arithmetic can be optimally implemented using the FPGA's specialized floating point DSP (Digital Signal Processing) hardware.

With this optimization, our FPGA implementation requires 4*m* DSPs to compute the complex floating point dot product. Thus, the matrix size is constrained by the total FPGA DSP resources available. Note that this upper bound is a consequence of this particular implementation.

By default, the design is parameterized to process 128 × 128 matrices when compiled targeting Intel® PAC with Intel Arria® 10 GX FPGA. It is parameterized to process 256 × 256 matrices when compiled targeting Intel® PAC with Intel Stratix® 10 SX FPGA, a larger device.
 

## Key Implementation Details
| Kernel            | Description
---                 |---
| QRD               | Implements a modified Gram-Schmidt QR decomposition algorithm. 

To optimize the performance-critical loop in its algorithm, the design leverages concepts discussed in the following FPGA tutorials: 
* **Triangular Loop Optimization** (triangular_loop)
* **Explicit Pipelining with `fpga_reg`** (fpga_register)
* **Loop `ivdep` Attribute** (loop_ivdep)
* **Unrolling Loops** (loop_unroll)

 The key optimization techniques used are as follows:
   1. Refactoring the algorithm to merge two dot products into one, reducing the total number of dot products needed to three from two. This helps us reduce the DSPs needed for the implementation.
   2. Converting the nested loop into a single merged loop and applying Triangular Loop optimizations. This allows us to generate a design that is very well pipelined.
   3. Fully vectorizing the dot products using loop unrolling.
   4. Using the compiler flag -Xsfp-relaxed to re-order floating point operations and allowing the inference of a specialised dot-product DSP. This further reduces the number of DSP blocks needed by the implementation, the overall latency, and pipeline depth.
   5. Using an efficient memory banking scheme to generate high performance hardware.
   6. Using the `fpga_reg` attribute to insert more pipeline stages where needed to improve the frequency achieved by the design.

## License  
This code sample is licensed under MIT license.

## Building the Reference Design

### Include Files
The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Code Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (fpga_compile or fpga_runtime) as well as whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/get-started/base-toolkit/](https://devcloud.intel.com/oneapi/get-started/base-toolkit/)).

When compiling for FPGA hardware, it is recommended to increase the job timeout to 24h.
 
### On a Linux* System
1. Install the design into a directory `build` from the design directory by running `cmake`:

   ```
   mkdir build
   cd build
   ```

   If you are compiling for the Intel® PAC with Intel Arria® 10 GX FPGA, run `cmake` using the command:

   ```
   cmake ..
   ```

   If instead you are compiling for the Intel® PAC with Intel Stratix® 10 SX FPGA, run `cmake` using the command:

   ```
   cmake .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
   ```

2. Compile the design through the generated `Makefile`. The following targets are provided and they match the recommended development flow:

    * Compile for emulation (fast compile time, targets emulated FPGA device).

       ```
       make fpga_emu
       ```

    * Generate HTML performance report. Find the report in `qrd_report.prj/reports/report.html`directory.

       ```
       make report
       ```

    * Compile for FPGA hardware (longer compile time, targets FPGA device).

       ```
       make fpga
       ```

3. (Optional) As the above hardware compile may take several hours to complete, an Intel® PAC with Intel Arria® 10 GX FPGA precompiled binary can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/qrd.fpga.tar.gz" download>here</a>.

### On a Windows* System
Note: `cmake` is not yet supported on Windows. A build.ninja file is provided instead. 

Note: Ensure that Microsoft Visual Studio* (2017, or 2019 Version 16.4 or newer) with "Desktop development with C++" workload is installed on your system.

1. Enter source file directory.

```
cd src
```

2. Compile the design. The following targets are provided and they match the recommended development flow:

    * Compile for emulation (fast compile time, targets emulated FPGA device).

      ```
      ninja fpga_emu
      ```

    * Generate HTML performance report. Find the report in `../src/qrd_report.prj/reports/report.html`directory.

      ```
      ninja report
      ```

      If you are targeting the Intel® PAC with Intel Stratix® 10 SX FPGA, please use the following target and find the report in `../src/qrd_s10_pac_report.prj/reports/report.html`.

      ```
      ninja report_s10_pac
      ```

    * **Not supported yet:** Compile and run on an FPGA hardware.

### In Third-Party Integrated Development Environments (IDEs)

You can compile and run this Reference Design in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)

## Running the Reference Design
You can apply QR decomposition to a number of matrices as shown below. This step performs the following:
* Generates the number of random matrices specified as the command line argument (defaults to 1).
* Computes QR decomposition on all matrices.
* Evaluates performance.
NOTE: The design is optimized to perform best when run on a large number of matrices, where the total number of matrices is a power of 2.



 1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
     ```
     ./qrd.fpga_emu           (Linux)
     qrd.fpga_emu.exe         (Windows)
     ```

2. Run the sample on the FPGA device. It is recommended to pass in an optional argument (as shown) when invoking the sample on hardware. Otherwise, the performance will not be representative.
     ```
     ./qrd.fpga 40960         (Linux)
     ```
### Application Parameters

| Argument | Description
---        |---
| `<num>`  | Optional argument that specifies the number of matrices to decompose. Its default value is `1`.

### Example of Output

Example output when running on Intel® PAC with Intel Arria® 10 GX FPGA for 32768 matrices (each of consisting of 128*128 complex numbers):

```
Device name: pac_a10 : Intel PAC Platform (pac_f000000)
Generating 32768 random matrices
Running QR decomposition of 32768 matrices repeatedly
   Total duration:   41.3763 s
Throughput: 25.3425k matrices/s
Verifying results on matrix 0 16384 32767
PASSED
```

Example output when running on Intel® PAC with Intel Stratix® 10 SX FPGA for 40960 matrices (each of consisting of 256*256 complex numbers):

```
Device name: pac_s10 : Intel PAC Platform (pac_f100000)
Generating 4096 random matrices
Running QR decomposition of 4096 matrices repeatedly
   Total duration:   17.3197 s
Throughput: 7.5678k matrices/s
Verifying results on matrix 0 2048 4095
PASSED
```

## Additional Design Information

### Compiler Flags Used

| Flag | Description
---    |---
`-Xshardware` | Target FPGA hardware (as opposed to FPGA emulator)
`-Xsclock=330MHz` | The FPGA backend attempts to achieve 330 MHz
`-Xsfp-relaxed` | Allows the FPGA backend to re-order floating point arithmetic operations (e.g. permit assuming (a + b + c) == (c + a + b) ) 
`-Xsparallel=2` | Use 2 cores when compiling the bitstream through Quartus
`-Xsseed` | Specifies the Quartus compile seed, to yield slightly higher fmax
`-DROWS_COMPONENT` | Specifies the number of rows of the matrix
`-DCOLS_COMPONENT` | Specifies the number of columns of the matrix
`-DFIXED_ITERATIONS` | Used to set the ivdep safelen attribute for the performance critical triangular loop

NOTE: The values for `seed`, `FIXED_ITERATIONS`, `ROWS_COMPONENT`, `COLS_COMPONENT` are set according to the board being targeted.

### Performance disclaimers

Tests document performance of components on a particular test, in specific systems. Differences in hardware, software, or configuration will affect actual performance. Consult other sources of information to evaluate performance as you consider your purchase.  For more complete information about performance and benchmark results, visit [www.intel.com/benchmarks](www.intel.com/benchmarks).

Performance results are based on testing as of July 29, 2020 and may not reflect all publicly available security updates.  See configuration disclosure for details.  No product or component can be absolutely secure.

Intel technologies’ features and benefits depend on system configuration and may require enabled hardware, software or service activation. Performance varies depending on system configuration. Check with your system manufacturer or retailer or learn more at [intel.com](www.intel.com).

The performance was measured by Intel on July 29, 2020.

Intel and the Intel logo are trademarks of Intel Corporation or its subsidiaries in the U.S. and/or other countries.

(C) Intel Corporation.
      

