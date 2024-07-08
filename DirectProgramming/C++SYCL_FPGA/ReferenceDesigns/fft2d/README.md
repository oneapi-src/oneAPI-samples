# `FFT2D` Sample
The `FFT2D` reference design demonstrates a Fast Fourier Transform (2D) implementation on FPGAs.


| Area                  | Description
|:---                   |:---
| What you will learn   | How to implement an FPGA version of the Fast Fourier Transform (2D).
| Time to complete      | 1 hr (not including compile time)
| Category              | Reference Designs and End to End

## Purpose

This FPGA reference design demonstrates the 2D Fast Fourier Transform (FFT) of complex matrices, a common linear algebra operation used in e.g. radar applications. 

## Prerequisites

This sample is part of the FPGA code samples.
It is categorized as a Tier 4 sample that demonstrates a reference design.

```mermaid
flowchart LR
   tier1("Tier 1: Get Started")
   tier2("Tier 2: Explore the Fundamentals")
   tier3("Tier 3: Explore the Advanced Techniques")
   tier4("Tier 4: Explore the Reference Designs")

   tier1 --> tier2 --> tier3 --> tier4

   style tier1 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier2 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier3 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier4 fill:#f96,stroke:#333,stroke-width:1px,color:#fff
```

Find more information about how to navigate this part of the code samples in the [FPGA top-level README.md](/DirectProgramming/C++SYCL_FPGA/README.md).
You can also find more information about [troubleshooting build errors](/DirectProgramming/C++SYCL_FPGA/README.md#troubleshooting), [running the sample on the Intel® DevCloud](/DirectProgramming/C++SYCL_FPGA/README.md#build-and-run-the-samples-on-intel-devcloud-optional), [using Visual Studio Code with the code samples](/DirectProgramming/C++SYCL_FPGA/README.md#use-visual-studio-code-vs-code-optional), [links to selected documentation](/DirectProgramming/C++SYCL_FPGA/README.md#documentation), etc.

| Optimized for        | Description
|:---                  |:---
| OS                   | Ubuntu* 20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10 <br> Windows Server* 2019
| Hardware             | Intel® Agilex® 7, Arria® 10, and Stratix® 10 FPGAs
| Software             | Intel® oneAPI DPC++/C++ Compiler

> **Note**: Even though the Intel DPC++/C++ oneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> For using the simulator flow, Intel® Quartus® Prime Pro Edition and one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition must be installed and accessible through your PATH.
>
> :warning: Make sure you add the device files associated with the FPGA that you are targeting to your Intel® Quartus® Prime installation.

## Key Implementation Details

The sample processes a 2D matrix of complex single-precision floating-point data. 
The 2D transform is decomposed as a 1D FFT applied to each row followed by a 1D FFT applied to each column.
The device portion of the code applies a 1D transform to each row and transposes the matrix. 
This computation is enqueued twice from the host code. 
The second iteration applies a 1D transform to each transposed row, producing the column-wise FFT. 
The second transposition restores the original matrix layout.
The 1D FFT engine is written as a single work-item kernel. 
To achieve efficient memory access two additional kernels are used to read / write data from / to global memory. 
All three kernels run concurrently and exchange data through kernel to kernel pipes.

The 1D FFT algorithm implements the 4-parallel and 8-parallel versions of the "Pipeline Radix-2k Feedforward FFT Architectures" paper written by Mario Garrido, Jesús Grajal, M. A. Sanchez, Oscar Gustafsson (published in IEEE Trans. VLSI Syst. 21(1): 23-32 (2013)).

To optimize the performance-critical parts of the algorithm, the design leverages concepts discussed in the following FPGA tutorials:

- **Unrolling Loops** (loop_unroll)
- **Explicit data movement** (explicit_data_movement)
- **Data Transfers Using Pipes** (pipes)

Additionally, the `cmake` build system can be configured using the following parameters:

| `cmake` option               | Description
|:---                          |:---
| `-DSET_PARALLELISM=[4/8]`    | Specifies if the 4-parallel or the 8-parallel version of the code should be implemented. Default is 8, except when running the simulator flow where 4 is being forced.
| `-DSET_LOGN=[N]`             | Specifies the log2 of one dimension of the 2D array. The resulting input to the 2D FFT will be (1 << N) * (1 << N). Default is 10, except when running the simulator flow where N is scaled back to 4.

## Build the `FFT2D` Sample

> **Note**: When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables.
> Set up your CLI environment by sourcing the `setvars` script located in the root of your oneAPI installation every time you open a new terminal window.
> This practice ensures that your compiler, libraries, and tools are ready for development.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\"Program Files (x86)"\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### On Linux*

1. Change to the sample directory.
2. Configure the build system for the Agilex® 7 device family, which is the default.

   ```
   mkdir build
   cd build
   cmake ..
   ```

   > **Note**: You can change the default target by using the command:
   >  ```
   >  cmake .. -DFPGA_DEVICE=<FPGA device family or FPGA part number>
   >  ```
   >
   > Alternatively, you can target an explicit FPGA board variant and BSP by using the following command:
   >  ```
   >  cmake .. -DFPGA_DEVICE=<board-support-package>:<board-variant> -DIS_BSP=1
   >  ```
  > **Note**: You can poll your system for available BSPs using the `aoc -list-boards` command. The board list that is printed out will be of the form
  > ```
  > $> aoc -list-boards
  > Board list:
  >   <board-variant>
  >      Board Package: <path/to/board/package>/board-support-package
  >   <board-variant2>
  >      Board Package: <path/to/board/package>/board-support-package
  > ```
   >
   > You will only be able to run an executable on the FPGA if you specified a BSP.

3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile for emulation (fast compile time, targets emulated FPGA device).
      ```
      make fpga_emu
      ```
   2. Compile for simulation (fast compile time, targets simulator FPGA device):
      ```
      make fpga_sim
      ```
   3. Generate HTML performance report.
      ```
      make report
      ```
      The report resides at `fft2d.report.prj/reports/report.html`.

   4. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      make fpga
      ```

### On Windows*

1. Change to the sample directory.
2. Configure the build system for the Agilex® 7 device family, which is the default.
   ```
   mkdir build
   cd build
   cmake -G "NMake Makefiles" ..
   ```

  > **Note**: You can change the default target by using the command:
  >  ```
  >  cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<FPGA device family or FPGA part number>
  >  ```
  >
  > Alternatively, you can target an explicit FPGA board variant and BSP by using the following command:
  >  ```
  >  cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<board-support-package>:<board-variant> -DIS_BSP=1
  >  ```
  > **Note**: You can poll your system for available BSPs using the `aoc -list-boards` command. The board list that is printed out will be of the form
  > ```
  > $> aoc -list-boards
  > Board list:
  >   <board-variant>
  >      Board Package: <path/to/board/package>/board-support-package
  >   <board-variant2>
  >      Board Package: <path/to/board/package>/board-support-package
  > ```
  >
  > You will only be able to run an executable on the FPGA if you specified a BSP.

3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile for emulation (fast compile time, targets emulated FPGA device).
      ```
      nmake fpga_emu
      ```
   2. Compile for simulation (fast compile time, targets simulator FPGA device):
      ```
      nmake fpga_sim
      ```
   3. Generate HTML performance report.
      ```
      nmake report
      ```
      The report resides at `fft2d_report.a.prj/reports/report.html`.

   4. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      nmake fpga
      ```
>**Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your 'build' directory in a shorter path, for example `C:\samples\build`. You can then run cmake from that directory, and provide cmake with the full path to your sample directory, for example:
>
>  ```
  > C:\samples\build> cmake -G "NMake Makefiles" C:\long\path\to\code\sample\CMakeLists.txt
>  ```
## Run the `FFT2D` Design

### Configurable Parameters

| Argument  | Description
|:---       |:---
| `<mode>`  | (Optional) Selects the mode to run between `normal`, `inverse`, `mangle`, `inverse-mangle` and `all` (the default mode if <mode> is omitted).


### On Linux

#### Run on FPGA Emulator

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./fft2d.fpga_emu
   ```

#### Run on FPGA Simulator

1. Run the sample on the FPGA simulator.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./fft2d.fpga_sim
   ```

#### Run on FPGA

1. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   ./fft2d.fpga
   ```

### On Windows

#### Run on FPGA Emulator

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   fft2d.fpga_emu.exe
   ```

#### Run on FPGA Simulator

1. Run the sample on the FPGA simulator.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   fft2d.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```

#### Run on FPGA

1. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   fft2d.fpga.exe
   ```

## Example Output


Example Output when running on the **Intel® FPGA SmartNIC N6001-PL**.

```
No program argument was passed, running all fft2d variants
Running on device: ofs_n6001 : Intel OFS Platform (ofs_ee00000)
Using USM device allocations
Launching a 1048576 points 8-parallel FFT transform (ordered data layout)
Processing time = 0.00187981s
Throughput = 0.55781 Gpoints / sec (55.781 Gflops)
Signal to noise ratio on output sample: 137.231
 --> PASSED
Running on device: ofs_n6001 : Intel OFS Platform (ofs_ee00000)
Using USM device allocations
Launching a 1048576 points 8-parallel inverse FFT transform (ordered data layout)
Processing time = 0.00184986s
Throughput = 0.566842 Gpoints / sec (56.6842 Gflops)
Signal to noise ratio on output sample: 136.861
 --> PASSED
Running on device: ofs_n6001 : Intel OFS Platform (ofs_ee00000)
Using USM device allocations
Launching a 1048576 points 8-parallel FFT transform (alternative data layout)
Processing time = 0.00185805s
Throughput = 0.564343 Gpoints / sec (56.4343 Gflops)
Signal to noise ratio on output sample: 137.436
 --> PASSED
Running on device: ofs_n6001 : Intel OFS Platform (ofs_ee00000)
Using USM device allocations
Launching a 1048576 points 8-parallel inverse FFT transform (alternative data layout)
Processing time = 0.00185293s
Throughput = 0.565902 Gpoints / sec (56.5902 Gflops)
Signal to noise ratio on output sample: 136.689
 --> PASSED
```

## License

Code samples are licensed under the MIT license. See [License.txt](/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](/third-party-programs.txt).
