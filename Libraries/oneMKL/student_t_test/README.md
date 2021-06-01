# `Student's T-test` Sample

The Student's T-test sample shows how to use the oneMKL library's Vector Statistics functionality to decide if the null hypothesis should be accepted or rejected.

For more information on oneMKL, and complete documentation of all oneMKL routines, see https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html.

| Optimized for       | Description
|:---                 |:---
| OS                  | Linux* Ubuntu* 18.04; Windows 10
| Hardware            | Skylake with Gen9 or newer
| Software            | Intel&reg; oneMKL
| What you will learn | How to use oneMKL's Vector Statistics
| Time to complete    | 15 minutes


## Purpose

Student’s t-test, in statistics, a method of testing hypotheses about the mean of a small sample drawn from a normally distributed population when the population standard deviation is unknown. It is usually first to formulate a null hypothesis, which states that there is no effective difference between the observed sample mean and the hypothesized or stated population mean—i.e., that any measured difference is due only to chance.

This sample uses oneMKL's Vector Statistics functionality to produce the random numbers and compute statistics.

The computations of the Student's t-test sample are performed on the default DPC++ device. You can set the `SYCL_DEVICE_TYPE` environment variable to `cpu` or `gpu` to select the device to use.


## Key Implementation Details

The student's t-test sample illustrates how to create an RNG engine object (the source of pseudo-randomness), a distribution object (specifying the desired probability distribution), and generate the random numbers themselves. After the numbers are produced, basic statistical properties such as mean and standard deviation are computed to be processed inside the Student's T-test algorithm.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)


## Building the Student's T-test Sample

# `Student's T-test` Sample

Student's T-test shows how to use the oneMKL library's Vector Statistics functionality to to decide if the null hypothesys should be accepted or rejected.

For more information on oneMKL, and complete documentation of all oneMKL routines, see https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html.

| Optimized for       | Description
|:---                 |:---
| OS                  | Linux* Ubuntu* 18.04; Windows 10
| Hardware            | Skylake with Gen9 or newer
| Software            | Intel&reg; oneMKL
| What you will learn | How to use oneMKL's Vector Statistics
| Time to complete    | 15 minutes


## Purpose

Student’s t-test, in statistics, a method of testing hypotheses about the mean of a small sample drawn from a normally distributed population when the population standard deviation is unknown. It is usual first to formulate a null hypothesis, which states that there is no effective difference between the observed sample mean and the hypothesized or stated population mean—i.e., that any measured difference is due only to chance.

This sample uses oneMKL's Vector Statistics functionality to produce the random numbers and compute statistics.

This sample performs its computations on the default DPC++ device. You can set the `SYCL_DEVICE_TYPE` environment variable to `cpu` or `gpu` to select the device to use.


## Key Implementation Details

This sample illustrates how to create an RNG engine object (the source of pseudo-randomness), a distribution object (specifying the desired probability distribution), and generate the random numbers themselves. After the numbers are produced basic statistical properties such as mean and standard deviation are cumputed to be processed inside Student's T-test algorithm.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)


## Building the Student's T-test Sample

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) as well whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Linux* System
Run `make` to build and run the sample. Two programs (t_test and t_test_usm) are generated, which illustrate different APIs for random number generation.

You can remove all generated files with `make clean`.

### On a Windows* System
Run `nmake` to build and run the sample. Two programs (t_test.exe and t_test_usm.exe) are generated, which illustrate different APIs for random number generation.

You can remove all generated files with `nmake clean`.

*Warning*: On Windows, static linking with oneMKL currently takes a very long time, due to a known compiler issue. This will be addressed in an upcoming release.
## Running the Student's T-test Sample

### Example of Output
If everything is working correctly, after running `make` (`nmake`) you will see step-by-step output from each of the two example programs, providing the decision about accepting null hypothesis.
```
./t_test

Student's T-test Simulation
Buffer Api
-------------------------------------
Number of random samples = 1000000 with mean = 0, std_dev = 1
T-test result with expected mean: 1
T-test result with two input arrays: 1

./t_test_usm

Student's T-test Simulation
Unified Shared Memory Api
-------------------------------------
Number of random samples = 1000000 with mean = 0, std_dev = 1
T-test result with expected mean: 1
T-test result with two input arrays: 1
```