# Intel® DPC++ Compatibility Tool: `Vector Add` Sample

This sample demonstrates how to migrate a simple program from CUDA to
SYCL-compliant code. Vector Add provides an easy way to verify that
your development environment is set up correctly to use the Intel® DPC++
Compatibility Tool (DPCT).

| Optimized for         | Description
|:---                   |:---
| OS                    | Linux* Ubuntu* 18.04 <br> Windows* 10
| Software              | Intel&reg; DPC++ Compatibility Tool
| What you will learn   | Simple invocation of dpct to migrate CUDA code
| Time to complete      | 10 minutes


## Purpose

This simple project adds two vectors of `[1..N]` and prints the result. It starts as a CUDA project to provide you with an example of migrating from an existing CUDA project to a SYCL-compliant project.

The migration of existing CUDA projects to SYCL-compliant projects may
result in warnings printed to the console and added as comments in
the migrated source, which will use .dp.cpp file extensions. Signs represent
areas in the resulting source code that require additional attention from the
developer. This is because the code could not be migrated by the tool or some
other reasons that need further review and manual work for the
code to be SYCL compliant, correct, or performant. For this sample,
the warning results from a difference in how the original code and generated
code handle errors.

## Key Implementation Details
In addition to verifying that the necessary tools and files are installed, please ensure that they are
configured correctly on your system. This sample shows the basic invocation and workflow for using dpct.

## Using Visual Studio Code* (Optional)
You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel&reg; oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel&reg; oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.
 - (Linux only) Debug your GPU application with GDB for Intel® oneAPI toolkits using the **Generate Launch Configurations** extension.

To learn more about the extensions, see the 
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).


## Migrating the CUDA* Sample to Data Parallel C++ with the Intel® DPC++ Compatibility Tool
Building and running the CUDA sample is not required to migrate this project
to a SYCL*-compliant project.

> **Note**: Certain CUDA header files, referenced by the CUDA application
> source files to be migrated, need to be accessible for the migration step.
> See *Before you Begin* in [Get Started with the Intel® DPC++ Compatibility Tool](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-intel-dpcpp-compatibility-tool/top.html#top_BEFORE_YOU_BEGIN).

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: `. ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `$ bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
> - For Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or MacOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### Command Line On a Linux* System
1. Use dpct to migrate the CUDA code. The  migrated source code will be
   created in a new directory, by default named `dpct_output`.
   The dpct `--in-root` option is used to set the root location of the program
   sources that are to be migrated. Only files and folders located within the
   --in-root directory will be considered for migration by the tool. Files located
   outside the`--in-root` directory are considered system files and will not be
   migrated, even if they are included by a source file located within the
   `--in-root`directory.

```sh
# From the repo root directory:
$ dpct --in-root=. src/vector_add.cu
```

2. Inspect the migrated source code, address any `DPCT` warnings generated
   by the Intel® DPC++ Compatibility Tool, and verify the new program correctness.

Warnings are printed to the console and added as comments in the migrated
source. See *Diagnostic Reference* in the [Intel® DPC++ Compatibility Tool Developer Guide and Reference](https://www.intel.com/content/www/us/en/develop/documentation/intel-dpcpp-compatibility-tool-user-guide/top/diagnostics-reference.html) for more information on what each warning means.

This sample should generate the following warning:

```
warning: DPCT1003:0: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
```

See the section titled **Addressing Warnings in Migrated Code** below to
understand how to resolve the warnings.

3. Copy the original `Makefile` into the `dpct_output` folder and update the
   copy to build the migrated project using SYCL. Replace the CUDA
   configurations in that new `Makefile` with the following for use with SYCL:

```make
CXX = dpcpp
TARGET = vector_add
SRCS = src/vector_add.dp.cpp

# The remainder of the Makefile should work without changes.
```
4. Switch to the migration directory with `cd dpct_output`

5. Build the migrated sample with the `make` command.

6. Run the migrated sample with the `make run` command. You should see a block
   of even numbers, indicating the result of adding two vectors:
   `[1..N] + [1..N]`.

7. Clean up the build with the `make clean` command.


## Microsoft Visual Studio on Windows

1. Open the migration wizard at `Extensions` > `Intel` > `Migrate Project to DPC++`
   and choose the `vector-add.vcxproj` project.

2. Configure and run the migration. Use the default settings to create a new
   project, which will be added to the open solution.

Notice the migrated command-line invocation. You can run this from the command
line as long as you first initialize your environment with:

```sh
"C:\Program Files (x86)\intel\oneapi\setvars.bat"
```

3. Inspect the migrated source code and address any `DPCT` warnings generated
   by the Intel® DPC++ Compatibility Tool. Warnings appear in a tool window and
   are written to a `migration.log` file in the project directory.

This sample should generate the following warning:

```
warning: DPCT1003:0: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
```
See below **Addressing Warnings in the Migrated Code** to understand how to resolve the warning.

4. Build and run the migrated project by right-clicking the project in the
   solution explorer, selecting it as the startup project and running it with
   the green play button in the top bar.


# Addressing Warnings in Migrated Code

Migration generated one warning for code that `dpct` could not migrate:

```
warning: DPCT1003:0: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
```

Since SYCL uses exceptions rather than error codes for error handling, the
tool removed the conditional statement to exit on failure and instead wrapped
the code in a `try` block. However, `dpct` retained the error status variable
and changed the source to always assign an error code of `0` to it. One way to
address the associated migration comment would be to remove the status
variable entirely.

To manually resolve the issue, simply remove the variable `status`, since it
is not needed.


# Example Output

When you run the migrated application, you should see console output, which
lists a group of even numbers produced by the kernel code's execution of
`((index+1) + (index+1))`.

```
./vector_add

  2   4   6   8  10  12  14  16  18  20  22  24  26  28  30  32
 34  36  38  40  42  44  46  48  50  52  54  56  58  60  62  64
 66  68  70  72  74  76  78  80  82  84  86  88  90  92  94  96
 98 100 102 104 106 108 110 112 114 116 118 120 122 124 126 128
130 132 134 136 138 140 142 144 146 148 150 152 154 156 158 160
162 164 166 168 170 172 174 176 178 180 182 184 186 188 190 192
194 196 198 200 202 204 206 208 210 212 214 216 218 220 222 224
226 228 230 232 234 236 238 240 242 244 246 248 250 252 254 256
258 260 262 264 266 268 270 272 274 276 278 280 282 284 286 288
290 292 294 296 298 300 302 304 306 308 310 312 314 316 318 320
322 324 326 328 330 332 334 336 338 340 342 344 346 348 350 352
354 356 358 360 362 364 366 368 370 372 374 376 378 380 382 384
386 388 390 392 394 396 398 400 402 404 406 408 410 412 414 416
418 420 422 424 426 428 430 432 434 436 438 440 442 444 446 448
450 452 454 456 458 460 462 464 466 468 470 472 474 476 478 480
482 484 486 488 490 492 494 496 498 500 502 504 506 508 510 512
```

If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).