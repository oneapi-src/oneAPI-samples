# QR Decomposition of Matrices
This DPC++ reference design demonstrates high performance QR decomposition of complex matrices on FPGA.

***Documentation***:  The [DPC++ FPGA Code Samples Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of DPC++ for FPGA. <br>
The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) is the reference manual for targeting FPGAs through DPC++. <br>
The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX) <br> Intel Xeon® CPU E5-1650 v2 @ 3.50GHz (host machine)
| Software                          | Intel® oneAPI DPC++ Compiler <br> Intel® FPGA Add-On for oneAPI Base Toolkit
| What you will learn               | Implementing a high performance FPGA version of the Gram-Schmidt QR decomposition algorithm.
| Time to complete                  | 1 hr (not including compile time)




**Performance**
Please refer to the performance disclaimer at the end of this README.

| Device                                         | Throughput
|:---                                            |:---
| Intel® PAC with Intel Arria® 10 GX FPGA        | 25k matrices/s for matrices of size 128 * 128
| Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX)      | 7k matrices/s for matrices of size 256 * 256


## Purpose

This FPGA reference design demonstrates QR decomposition of matrices of complex numbers, a common operation employed in linear algebra. Matrix _A_ (input) is decomposed into a product of an orthogonal matrix _Q_ and an upper triangular matrix _R_.

The algorithms employed by the reference design are the Gram-Schmidt QR decomposition algorithm and the thin QR factorization method. Background information on these algorithms can be found in Wikipedia's [QR decomposition](https://en.wikipedia.org/wiki/QR_decomposition) article. The original algorithm has been modified and optimized for performance on FPGAs in this implementation.

QR decomposition is used extensively in signal processing applications such as beamforming, multiple-input multiple-output (MIMO) processing, and Space Time Adaptive Processing (STAP).


### Matrix dimensions and FPGA resources

The QR decomposition algorithm factors a complex _m_×_n_ matrix, where _m_ ≥ _n_. The algorithm computes the vector dot product of two columns of the matrix. In our FPGA implementation, the dot product is computed in a loop over the column's _m_ elements. The loop is fully unrolled to maximize throughput. As a result, *m* complex multiplication operations are performed in parallel on the FPGA, followed by sequential additions to compute the dot product result. 

We use the compiler flag `-fp-relaxed`, which permits the compiler to reorder floating point additions (i.e. to assume that floating point addition is commutative). The compiler uses this freedom to reorder the additions so that the dot product arithmetic can be optimally implemented using the FPGA's specialized floating point DSP (Digital Signal Processing) hardware.

Note: the compiler flag '-fp-relaxed' will be deprecated in the next release and replaced by a new implementation.

With this optimization, our FPGA implementation requires 4*m* DSPs to compute the complex floating point dot product. Thus, the matrix size is constrained by the total FPGA DSP resources available. Note that this upper bound is a consequence of this particular implementation.

By default, the design is parameterized to process 128 × 128 matrices when compiled targeting Intel® PAC with Intel Arria® 10 GX FPGA. It is parameterized to process 256 × 256 matrices when compiled targeting Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX), a larger device.
 

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
   1. Refactoring the algorithm to merge two dot products into one, reducing the total number of dot products needed to three from two. This helps us reduce the DSPs required for the implementation.
   2. Converting the nested loop into a single merged loop and applying Triangular Loop optimizations. This allows us to generate a design that is very well pipelined.
   3. Fully vectorizing the dot products using loop unrolling.
   4. Using the compiler flag -Xsfp-relaxed to re-order floating point operations and allowing the inference of a specialised dot-product DSP. This further reduces the number of DSP blocks needed by the implementation, the overall latency, and pipeline depth.
   5. Using an efficient memory banking scheme to generate high performance hardware.
   6. Using the `fpga_reg` attribute to insert more pipeline stages where needed to improve the frequency achieved by the design.

## License  
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the Reference Design

### Include Files
The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Code Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (fpga_compile, fpga_runtime:arria10, or fpga_runtime:stratix10) and whether to run in batch or interactive mode. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/documentation/base-toolkit/](https://devcloud.intel.com/oneapi/documentation/base-toolkit/)).

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

   If instead you are compiling for the Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX), run `cmake` using the command:

   ```
   cmake .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
   ```

2. Compile the design through the generated `Makefile`. The following targets are provided, and they match the recommended development flow:

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

3. (Optional) As the above hardware compile may take several hours to complete, FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/qrd.fpga.tar.gz" download>here</a>.

### On a Windows* System
1. Generate the `Makefile` by running `cmake`.
     ```
   mkdir build
   cd build
   ```
   To compile for the Intel® PAC with Intel Arria® 10 GX FPGA, run `cmake` using the command:  
    ```
    cmake -G "NMake Makefiles" ..
   ```
   Alternatively, to compile for the Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX), run `cmake` using the command:
   ```
   cmake -G "NMake Makefiles" .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
   ```

2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:
   * Compile for emulation (fast compile time, targets emulated FPGA device): 
     ```
     nmake fpga_emu
     ```
   * Generate the optimization report: 
     ```
     nmake report
     ``` 
   * An FPGA hardware target is not provided on Windows*. 

*Note:* The Intel® PAC with Intel Arria® 10 GX FPGA and Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) do not yet support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.

### In Third-Party Integrated Development Environments (IDEs)

You can compile and run this Reference Design in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)

## Running the Reference Design
You can apply QR decomposition to a number of matrices, as shown below. This step performs the following:
* Generates the number of random matrices specified as the command line argument (defaults to 1).
* Computes QR decomposition on all matrices.
* Evaluates performance.
NOTE: The design is optimized to perform best when run on a large number of matrices, where the total number of matrices is a power of 2.


 1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
 Increase the amount of memory that the emulator runtime is permitted to allocate by setting the CL_CONFIG_CPU_FORCE_PRIVATE_MEM_SIZE environment variable before running the executable.
     ```
     export CL_CONFIG_CPU_FORCE_PRIVATE_MEM_SIZE=32MB
     ./qrd.fpga_emu           (Linux)
 
     set CL_CONFIG_CPU_FORCE_PRIVATE_MEM_SIZE=32MB
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

Example output when running on Intel® PAC with Intel Arria® 10 GX FPGA for 32768 matrices (each consisting of 128*128 complex numbers):

```
Device name: pac_a10 : Intel PAC Platform (pac_f000000)
Generating 32768 random matrices
Running QR decomposition of 32768 matrices repeatedly
   Total duration:   41.3763 s
Throughput: 25.3425k matrices/s
Verifying results on matrix 0 16384 32767
PASSED
```

Example output when running on Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) for 40960 matrices (each consisting of 256*256 complex numbers):

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

### Host Limitations
The QRD demo host is not optimized for a very large number of matrices. Running the QRD executable with number of matrices that occupy more memory than what is physically available on the host machine will result in system performance degradation due to virtual memory thrashing by the operating system.

### Performance disclaimers

Tests document performance of components on a particular test, in specific systems. Differences in hardware, software, or configuration will affect actual performance. Consult other sources of information to evaluate performance as you consider your purchase.  For more complete information about performance and benchmark results, visit [www.intel.com/benchmarks](www.intel.com/benchmarks).

Performance results are based on testing as of July 29, 2020 and may not reflect all publicly available security updates.  See configuration disclosure for details.  No product or component can be absolutely secure.

Intel technologies’ features and benefits depend on system configuration and may require enabled hardware, software or service activation. Performance varies depending on system configuration. Check with your system manufacturer or retailer or learn more at [intel.com](www.intel.com).

The performance was measured by Intel on July 29, 2020.

Intel and the Intel logo are trademarks of Intel Corporation or its subsidiaries in the U.S. and/or other countries.

(C) Intel Corporation.
      

