# QR Decomposition of Matrices
This reference design demonstrates high performance QR decomposition of complex/real matrices on FPGA.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04/20.04 <br>RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware                          | Intel&reg; Programmable Acceleration Card (PAC) with Intel Arria&reg; 10 GX FPGA <br> Intel&reg; FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix&reg; 10 SX) <br> Intel Xeon&reg; CPU E5-1650 v2 @ 3.50GHz (host machine)
| Software                          | Intel&reg; oneAPI DPC++ Compiler <br> Intel&reg; FPGA Add-On for oneAPI Base Toolkit
| What you will learn               | Implementing a high performance FPGA version of the Gram-Schmidt QR decomposition algorithm.
| Time to complete                  | 1 hr (not including compile time)


**Performance** 

Please refer to the performance disclaimer at the end of this README.

| Device                                         | Throughput
|:---                                            |:---
| Intel&reg; PAC with Intel Arria&reg; 10 GX FPGA        | 24k matrices/s for complex matrices of size 128 * 128
| Intel&reg; FPGA PAC D5005 (with Intel Stratix&reg; 10 SX)      | 7k matrices/s for complex matrices of size 256 * 256


## Purpose

This FPGA reference design demonstrates QR decomposition of matrices of complex/real numbers, a common operation employed in linear algebra. Matrix _A_ (input) is decomposed into a product of an orthogonal matrix _Q_ and an upper triangular matrix _R_.

The algorithms employed by the reference design are the Gram-Schmidt QR decomposition algorithm and the thin QR factorization method. Background information on these algorithms can be found in Wikipedia's [QR decomposition](https://en.wikipedia.org/wiki/QR_decomposition) article. The original algorithm has been modified and optimized for performance on FPGAs in this implementation.

QR decomposition is used extensively in signal processing applications such as beamforming, multiple-input multiple-output (MIMO) processing, and Space Time Adaptive Processing (STAP).


### Additional Documentation
- [Explore SYCL* Through Intel&reg; FPGA Code Samples](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of FPGAs and SYCL.
- [FPGA Optimization Guide for Intel&reg; oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) helps you understand how to target FPGAs using SYCL and Intel&reg; oneAPI Toolkits.
- [Intel&reg; oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) helps you understand target-independent, SYCL-compliant programming using Intel&reg; oneAPI Toolkits.

### Matrix dimensions and FPGA resources

The QR decomposition algorithm factors a complex _m_ × _n_ matrix, where _m_ ≥ _n_. The algorithm computes the vector dot product of two columns of the matrix. In our FPGA implementation, the dot product is computed in a loop over the column's _m_ elements. The loop is fully unrolled to maximize throughput. As a result, *m* complex multiplication operations are performed in parallel on the FPGA, followed by sequential additions to compute the dot product result.

We use the compiler flag `-fp-relaxed`, which permits the compiler to reorder floating point additions (i.e. to assume that floating point addition is commutative). The compiler uses this freedom to reorder the additions so that the dot product arithmetic can be optimally implemented using the FPGA's specialized floating point DSP (Digital Signal Processing) hardware.

With this optimization, our FPGA implementation requires 4*_m_ DSPs to compute the complex floating point dot product or 2*m* DSPs for the real case. Thus, the matrix size is constrained by the total FPGA DSP resources available.

By default, the design is parameterized to process 128 × 128 matrices when compiled targeting Intel&reg; PAC with Intel Arria&reg; 10 GX FPGA. It is parameterized to process 256 × 256 matrices when compiled targeting Intel&reg; FPGA PAC D5005 (with Intel Stratix&reg; 10 SX), a larger device. However, the design can process matrices from 4 x 4 to 512 x 512.

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
   1. Refactoring the original Gram-Schmidt algorithm to merge two dot products into one, reducing the total number of dot products needed to three from two. This helps us reduce the DSPs required for the implementation.
   2. Converting the nested loop into a single merged loop and applying Triangular Loop optimizations. This allows us to generate a design that is very well pipelined.
   3. Fully vectorizing the dot products using loop unrolling.
   4. Using the compiler flag -Xsfp-relaxed to re-order floating point operations and allowing the inference of a specialized dot-product DSP. This further reduces the number of DSP blocks needed by the implementation, the overall latency, and pipeline depth.
   5. Using an efficient memory banking scheme to generate high performance hardware.
   6. Using the `fpga_reg` attribute to insert more pipeline stages where needed to improve the frequency achieved by the design.

## Building the Reference Design

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: `. ~/intel/oneapi/setvars.sh`
>
> Windows*:
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
>
>For more information on environment variables, see **Use the setvars Script** for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).


### Include Files
The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Code Samples in Intel&reg; DevCloud
If running a sample in the Intel&reg; DevCloud, remember that you must specify the type of compute node and whether to run in batch or interactive mode. Compiles to FPGA are only supported on fpga_compile nodes. Executing programs on FPGA hardware is only supported on fpga_runtime nodes of the appropriate type, such as fpga_runtime:arria10 or fpga_runtime:stratix10.  Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the Intel&reg; oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/documentation/base-toolkit/](https://devcloud.intel.com/oneapi/documentation/base-toolkit/)).

When compiling for FPGA hardware, it is recommended to increase the job timeout to 24h.


### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the [Using Visual Studio Code with Intel&reg; oneAPI Toolkits User Guide](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

### On a Linux* System
1. Install the design into a directory `build` from the design directory by running `cmake`:

   ```
   mkdir build
   cd build
   ```

   If you are compiling for the Intel&reg; PAC with Intel Arria&reg; 10 GX FPGA, run `cmake` using the command:

   ```
   cmake ..
   ```

   If instead you are compiling for the Intel&reg; FPGA PAC D5005 (with Intel Stratix&reg; 10 SX), run `cmake` using the command:

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
   To compile for the Intel&reg; PAC with Intel Arria&reg; 10 GX FPGA, run `cmake` using the command:
    ```
    cmake -G "NMake Makefiles" ..
   ```
   Alternatively, to compile for the Intel&reg; FPGA PAC D5005 (with Intel Stratix&reg; 10 SX), run `cmake` using the command:
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
   * Compile for FPGA hardware (longer compile time, targets FPGA device):
     ```
     nmake fpga
     ```

*Note:* The Intel&reg; PAC with Intel Arria&reg; 10 GX FPGA and Intel&reg; FPGA PAC D5005 (with Intel Stratix&reg; 10 SX) do not yet support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.<br>

*Note:* If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build. You can then run cmake from that directory, and provide cmake with the full path to your sample directory.


### Troubleshooting
If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
``make VERBOSE=1``
For more comprehensive troubleshooting, use the Diagnostics Utility for
Intel&reg; oneAPI Toolkits, which provides system checks to find missing
dependencies and permissions errors.
[Learn more](https://software.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

### In Third-Party Integrated Development Environments (IDEs)

You can compile and run this Reference Design in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [FPGA Workflows on Third-Party IDEs for Intel&reg; oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-oneapi-dpcpp-fpga-workflow-on-ide.html)

## Running the Reference Design
You can perform the QR decomposition of 8 matrices repeatedly, as shown below. This step performs the following:
* Generates 8 random matrices.
* Computes the QR decomposition of the 8 matrices.
* Repeats the decomposition multiple times (specified as a command line argument) to evaluate performance.


 1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
 Increase the amount of memory that the emulator runtime is permitted to allocate by setting the CL_CONFIG_CPU_FORCE_PRIVATE_MEM_SIZE environment variable before running the executable.
     ```
     export CL_CONFIG_CPU_FORCE_PRIVATE_MEM_SIZE=32MB
     ./qrd.fpga_emu           (Linux)

     set CL_CONFIG_CPU_FORCE_PRIVATE_MEM_SIZE=32MB
     qrd.fpga_emu.exe         (Windows)
     ```

2. Run the sample on the FPGA device.
     ```
     ./qrd.fpga         (Linux)
     qrd.fpga.exe       (Windows)
     ```
### Application Parameters

| Argument | Description
---        |---
| `<num>`  | Optional argument that specifies the number of times to repeat the decomposition of 8 matrices. Its default value is `16` for the emulation flow and '819200' for the FPGA flow.

### Example of Output

Example output when running on Intel&reg; PAC with Intel Arria&reg; 10 GX FPGA for 8 matrices 819200 times (each matrix consisting of 128*128 complex numbers):

```
Device name: pac_a10 : Intel PAC Platform (pac_f000000)
Generating 8 random complex matrices of size 128x128 
Running QR decomposition of 8 matrices 819200 times
 Total duration:   268.733 s
Throughput: 24.387k matrices/s
Verifying results...
PASSED
```

Example output when running on Intel&reg; FPGA PAC D5005 (with Intel Stratix&reg; 10 SX) for the decomposition of 8 matrices 819200 times (each matrix consisting of 256*256 complex numbers):

```
Device name: pac_s10 : Intel PAC Platform (pac_f100000)
Generating 8 random complex matrices of size 256x256 
Running QR decomposition of 8 matrices 819200 times
 Total duration:   888.077 s
Throughput: 7.37954k matrices/s
Verifying results...
PASSED
```

## Additional Design Information

### Compiler Flags Used

| Flag | Description
---    |---
`-Xshardware` | Target FPGA hardware (as opposed to FPGA emulator)
`-Xsclock=360MHz` | The FPGA backend attempts to achieve 360 MHz
`-Xsfp-relaxed` | Allows the FPGA backend to re-order floating point arithmetic operations (e.g. permit assuming (a + b + c) == (c + a + b) )
`-Xsparallel=2` | Use 2 cores when compiling the bitstream through Quartus
`-Xsseed` | Specifies the Quartus compile seed, to yield slightly higher fmax
`-DROWS_COMPONENT` | Specifies the number of rows of the matrix
`-DCOLS_COMPONENT` | Specifies the number of columns of the matrix
`-DFIXED_ITERATIONS` | Used to set the ivdep safelen attribute for the performance critical triangular loop
`-DCOMPLEX` | Used to select between the complex and real QR decomposition (complex is the default)

NOTE: The values for `seed`, `FIXED_ITERATIONS`, `ROWS_COMPONENT`, `COLS_COMPONENT` are set according to the board being targeted.

### Performance disclaimers

Tests document performance of components on a particular test, in specific systems. Differences in hardware, software, or configuration will affect actual performance. Consult other sources of information to evaluate performance as you consider your purchase.  For more complete information about performance and benchmark results, visit [this page](https://edc.intel.com/content/www/us/en/products/performance/benchmarks/overview).

Performance results are based on testing as of July 29, 2020 and may not reflect all publicly available security updates.  See configuration disclosure for details.  No product or component can be absolutely secure.

Intel technologies’ features and benefits depend on system configuration and may require enabled hardware, software or service activation. Performance varies depending on system configuration. Check with your system manufacturer or retailer or learn more at [intel.com](www.intel.com).

The performance was measured by Intel on July 29, 2020.

Intel and the Intel logo are trademarks of Intel Corporation or its subsidiaries in the U.S. and/or other countries.

&copy; Intel Corporation.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).