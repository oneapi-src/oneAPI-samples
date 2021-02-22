# CRR Binomial Tree Model for Option Pricing
An FPGA-optimized reference design computing the Cox-Ross-Rubinstein (CRR) binomial tree model with Greeks for American exercise options.

***Documentation***:  The [DPC++ FPGA Code Samples Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of DPC++ for FPGA. <br>
The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) is the reference manual for targeting FPGAs through DPC++. <br>
The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming. <br>
Additional reference material specific to option pricing algorithms is provided in the References section of this README.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA; <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX)
| Software                          | Intel® oneAPI DPC++ Compiler <br> Intel® FPGA Add-On for oneAPI Base Toolkit
| What you will learn               | Review a high performance DPC++ design optimized for FPGA
| Time to complete                  | 1 hr (not including compile time)




**Performance**
Please refer to the performance disclaimer at the end of this README.

| Device                                         | Throughput
|:---                                            |:---
| Intel® PAC with Intel Arria® 10 GX FPGA        | 118 assets/s
| Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX)      | 243 assets/s


## Purpose
This sample implements the Cox-Ross-Rubinstein (CRR) binomial tree model that is used in the finance field for American exercise options with five Greeks (delta, gamma, theta, vega and rho). The simple idea is to model all possible asset price paths using a binomial tree.

## Key Implementation Details

### Design Inputs
This design reads inputs from the `ordered_inputs.csv` file. The inputs are:

| Input                             | Description
---                                 |---
| `n_steps`                         | Number of time steps in the binomial tree. The maximum `n_steps` in this design is 8189.
| `cp`                              | -1 or 1 represents put and call options, respectively.
| `spot`                            | Spot price of the underlying price.
| `fwd`                             | Forward price of the underlying price.
| `strike`                          | Exercise price of the option.
| `vol`                             | Percent volatility that the design reads as a decimal value.
| `df`                              | Discount factor to option expiry.
| `t`                               | Time, in years, to the maturity of the option.

### Design Outputs
This design writes outputs to the `ordered_outputs.csv` file. The outputs are:

| Output                            | Description
---                                 |---
| `value`                           | Option price
| `delta`                           | Measures the rate of change of the theoretical option value with respect to changes in the underlying asset's price.
| `gamma`                           | Measures the rate of change in the `delta` with respect to changes in the underlying price.
| `vega`                            | Measures sensitivity to volatility.
| `theta`                           | Measures the sensitivity of the derivative's value to the passage of time.
| `rho`                             | Measures sensitivity to the interest of rate.

### Design Correctness
This design tests the optimized FPGA code's correctness by comparing its output to a golden result computed on the CPU.

### Design Performance
This design measures the FPGA performance to determine how many assets can be processed per second.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the CRR Program 

### Include Files
The included header `dpc_common.hpp` is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (fpga_compile, fpga_runtime:arria10, or fpga_runtime:stratix10) and whether to run in batch or interactive mode. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/documentation/base-toolkit/](https://devcloud.intel.com/oneapi/documentation/base-toolkit/)).

When compiling for FPGA hardware, it is recommended to increase the job timeout to 48h.

### On a Linux* System

1. Generate the `Makefile` by running `cmake`.
     ```
   mkdir build
   cd build
   ```
   To compile for the Intel® PAC with Intel Arria® 10 GX FPGA, run `cmake` using the command:  
    ```
    cmake ..
   ```
   Alternatively, to compile for the Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX), run `cmake` using the command:
 
   ```
   cmake .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
   ```
 
2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:
 
   * Compile for emulation (fast compile time, targets emulated FPGA device): 
      ```
      make fpga_emu
      ```
   * Generate the optimization report: 
     ```
     make report
     ``` 
   * Compile for FPGA hardware (longer compile time, targets FPGA device): 
     ```
     make fpga
     ``` 
3. (Optional) As the above hardware compile may take several hours to complete, FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/crr.fpga.tar.gz" download>here</a>.

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
 
You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)

## Running the Reference Design

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./crr.fpga_emu <input_file> [-o=<output_file>]                           (Linux)

     crr.fpga_emu.exe <input_file> [-o=<output_file>]                         (Windows)
     ```
 2. Run the sample on the FPGA device:
     ```
     ./crr.fpga <input_file> [-o=<output_file>]                               (Linux)
     ```

### Application Parameters

| Argument                          | Description
---                                 |---
| `<input_file>`                    | Optional argument that provides the input data. The default file is `/data/ordered_inputs.csv`
| `-o=<output_file>`                | Optional argument that specifies the name of the output file. The default name of the output file is `ordered_outputs.csv`.

### Example of Output
```
============ Correctness Test =============
Running analytical correctness checks...
CPU-FPGA Equivalence: PASS

============ Throughput Test =============
Avg throughput: 66.2 assets/s
```

## Additional Design Information

### Source Code Explanation

| File                              | Description
---                                 |---
| `main.cpp`                        | Contains both host code and SYCL* kernel code.
| `CRR_common.hpp`                  | Header file for `main.cpp`. Contains the data structures needed for both host code and SYCL* kernel code.

  

### Backend Compiler Flags Used

| Flag                              | Description
---                                 |---
`-Xshardware`                       | Target FPGA hardware (as opposed to FPGA emulator)
`-Xsdaz`                            | Denormals are zero
`-Xsrounding=faithful`              | Rounds results to either the upper or lower nearest single-precision numbers
`-Xsparallel=2`                     | Uses 2 cores when compiling the bitstream through Quartus
`-Xsseed=2`                         | Uses seed 2 during Quartus, yields slightly higher f<sub>MAX</sub>

### Preprocessor Define Flags 

| Flag                              | Description
---                                 |---
`-DOUTER_UNROLL=1`                  | Uses the value 1 for the constant OUTER_UNROLL, controls the number of CRRs that can be processed in parallel
`-DINNER_UNROLL=64`                 | Uses the value 64 for the constant INNER_UNROLL, controls the degree of parallelization within the calculation of 1 CRR
`-DOUTER_UNROLL_POW2=1`             | Uses the value 1 for the constant OUTER_UNROLL_POW2, controls the number of memory banks


NOTE: The Xsseed, DOUTER_UNROLL, DINNER_UNROLL and DOUTER_UNROLL_POW2 values differ depending on the board being targeted. More information about the unroll factors can be found in `/src/CRR_common.hpp`.

### Performance disclaimers

Tests document performance of components on a particular test, in specific systems. Differences in hardware, software, or configuration will affect actual performance. Consult other sources of information to evaluate performance as you consider your purchase. For more complete information about performance and benchmark results, visit [www.intel.com/benchmarks](www.intel.com/benchmarks).

Performance results are based on testing as of July 20, 2020 and may not reflect all publicly available security updates. See configuration disclosure for details. No product or component can be absolutely secure.

Intel technologies’ features and benefits depend on system configuration and may require enabled hardware, software or service activation. Performance varies depending on system configuration. Check with your system manufacturer or retailer or learn more at [intel.com](www.intel.com).

The performance was measured by Intel on July 20, 2020

Intel and the Intel logo are trademarks of Intel Corporation or its subsidiaries in the U.S. and/or other countries.

(C) Intel Corporation.

### References

[Khronous SYCL Resources](https://www.khronos.org/sycl/resources)

[Binomial options pricing model](https://en.wikipedia.org/wiki/Binomial_options_pricing_model)

[Wike page for finance Greeks](https://en.wikipedia.org/wiki/Greeks_(finance))

[OpenCL Intercept Layer](https://github.com/intel/opencl-intercept-layer)  

