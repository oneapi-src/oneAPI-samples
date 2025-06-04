# Fourier Correlation Sample
The cross-correlation has many applications, *e.g.*, measuring the similarity of two one-dimensional signals, finding the best translation to overlay similar images, volumetric medical image segmentation, etc. This sample shows how to implement one-dimensional and two-dimensional cross-correlations using SYCL, and oneMKL Discrete Fourier Transform (DFT) functions. This samples requires oneMKL 2024.1 (or newer).

For more information on oneMKL, and complete documentation of all oneMKL routines, see https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html.

For more information on supported systems and the corresponding requirements, see https://www.intel.com/content/www/us/en/developer/articles/system-requirements/intel-oneapi-base-toolkit-system-requirements.html.

## Purpose
This sample shows how to find the optimal translational shift maximizing the cross-correlation between two real, periodic signals $u$ and $v$ in ${&#8477;}^{n_{1}\times n_{2} \times \ldots \times n_{d}}$, *i.e.*, the (integer) value(s) $s_{i}, \ i \in  \lbrace 1, \ldots, d\rbrace$ maximizing

$$ c_{s_{1}, \ldots, s_{d-1}} = \sum_{j_{1} = 0}^{n_{1} - 1} \cdots \sum_{j_{d} = 0}^{n_{d} - 1} u_{j_{1}, \ldots, j_{d}} v_{j_{1} - s_{1}, \ldots, j_{d} - s_{d}}.$$

> **_NOTE:_** Given the periodic nature of $u$, $u_{j_{1} + k_{1} n_{1}, j_{2} + k_{2} n_{2}, \ldots, j_{d} + k_{d} n_{d}}$ and $u_{j_{1}, j_{2}, \ldots, j_{d}}$ are equal $\forall \\lbrace k_{1}, \ldots, k_{d}\rbrace \in {&#8484;}^{d}$; the same remark holds for $v$.

Discrete Fourier transforms may be used to evaluate the above $c$ efficiently, via

$$ c = \dfrac{1}{\prod_{i = 1}^{d} n_{i}} {&#120021;}^{-1} \left({&#120021;}(u) \odot \left({&#120021;}(v)\right)^{*}\right) $$

where ${&#120021;}$ (resp. ${&#120021;}^{-1}$) represents the forward (resp. backward) unscaled Discrete Fourier Transform, $\odot$ represents a component-wise product, and $\lambda^{*}$ is the complex conjugate of $\lambda$.

The implementations use SYCL and oneMKL. SYCL provides the device offload and host-device memory transfer mechanisms. oneMKL provides interfaces to compute forward and backward transforms on the device as well as other required functions like the component-wise product of two complex sequences. The result of the operation described above is compared with a naive SYCL implementation and the two are verified to be within numerical tolerance of each other.

## Implementation Details

In this sample, two artificial signals are created on the device. Their cross-correlation is evaluate 1) via a naive SYCL implementation and 2) by using the DFT-based procedure described above. The host retrieves the results and verifies that they are within numerical tolerance of each other. The optimal shift maximizing the cross-correlation between the signals is also extracted on the host, and reported as a normalized correlation score

$$ \rho_{u, v} = \dfrac{1}{\sigma_{u}\sigma_{v}} \left( \max_{\lbrace s_{1}, \ldots, s_{d} \rbrace} \dfrac{c_{s_{1}, \ldots, s_{d}}}{\prod_{i = i}^{d}{n_{i}}} - \overline{u}\overline{v} \right)$$

where $\overline{x}$ and $\sigma_{x}$ are the average value and standard deviations of $x$, respectively.

Two implementations of the one-dimensional algorithm are provided: one that uses explicit buffering and one that uses Unified Shared Memory (USM). Both implementations compute the cross-correlation on the selected device. A two-dimensional Fourier correlation example using USM is also included, illustrating how to define and use a two-dimensional data layout compliant with the requirements for in-place real-to-complex and complex-to-real transforms.

## License
Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Using Visual Studio Code* (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel Software Developer Tools**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel Software Developer Tools**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.
 - (Linux only) Debug your GPU application with GDB for Intel® oneAPI toolkits using the **Generate Launch Configurations** extension.

To learn more about the extensions, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

## Building and Running the Fourier Correlation Sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux Sudo: . /opt/intel/oneapi/setvars.sh
>
> Linux User: . ~/intel/oneapi/setvars.sh
>
> Windows: C:\"Program Files (x86)"\Intel\oneAPI\setvars.bat
>

>For more information on environment variables, see Use the setvars Script for [Linux or macOS](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2023-1/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2023-1/use-the-setvars-script-with-windows.html).

### On a Linux System
Run `make` to build and run the sample. One two-dimensional program (using USM) and two one-dimensional programs (one that uses explicit buffering and one that uses USM) are created.

You can remove all generated files with `make clean`.

### On a Windows System
Run `nmake` to build and run the sample.

Note: To remove temporary files, run `nmake clean`.

### Example of Output
The one-dimensional programs generate two artificial one-dimensional signals, computes their cross-correlation, and report the optimal (right-)shift for the second signal maximizing its correlation score with the first. The output should be similar to this:
```
./fcorr_1d_buff 4096
Running on: Intel(R) Data Center GPU Max 1550
Right-shift the second signal 2048 elements to get a maximum, normalized correlation score of 1 (treating the signals as periodic).
Max difference between naive and Fourier-based calculations : 2.38419e-07 (verification threshold: 6.66459e-06).
./fcorr_1d_usm 4096
Running on: Intel(R) Data Center GPU Max 1550
Right-shift the second signal 2048 elements to get a maximum, normalized correlation score of 1 (treating the signals as periodic).
Max difference between naive and Fourier-based calculations : 2.38419e-07 (verification threshold: 6.66459e-06).
```
For the two-dimensional case, the program generates two artificial two-dimensional images, computes their cross-correlation, and report the optimal translational vector for the second image maximizing its correlation score with the first. The output should be similar to this:

```
./fcorr_2d_usm
Running on: Intel(R) Data Center GPU Max 1550
First image:
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 1 1 0
0 0 0 0 0 1 1 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
Second image:
0 0 0 0 0 0 0 0
0 1 1 0 0 0 0 0
0 1 1 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
Shift the second signal by translation vector (3, 4) to get a maximum, normalized correlation score of 1 (treating the signals as periodic along both dimensions).
Max difference between naive and Fourier-based calculations : 1.19209e-07 (verification threshold: 4.91989e-06).
```

### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/docs/oneapi/user-guide-diagnostic-utility/current/overview.html).

## License

Code samples are licensed under the MIT license. See
[License.txt](License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](third-party-programs.txt).
