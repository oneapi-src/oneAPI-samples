# `Discrete Cosine Transform` Sample
Discrete Cosine Transform (DCT) and Quantization are the first two steps in the JPEG compression standard. The `Discrete Cosine Transform` sample demonstrates how DCT and Quantizing stages can be implemented to run faster using SYCL* by offloading image processing work to a GPU or other device.

| Area                     | Description
|:---                      |:---
| What you will learn      | How to parallel process image data with the Discrete Cosine Transform algorithm
| Time to complete         | 15 minutes

## Purpose
Discrete Cosine Transform (DCT) is a lossy compression algorithm used to represent every data point value using the sum of cosine functions, which are linearly orthogonal.

The sample program shows the possible effect of quality reduction in the image when one performs DCT, followed by quantization as found in JPEG compression.

This program generates an output image by first processing an input image using DCT and quantization, undoing the process through Inverse DCT and de-quantizing, and then writing the output to a BMP image file. The processing stage is performed on **8x8** subsections of the pixel image, referred to as 'blocks' in the code sample.

Since individual blocks of data can be processed independently, the overall image can be decomposed and processed in parallel, where each block represents a work item. Using SYCL*, parallelization can be implemented relatively quickly and with few changes to a serial version. The blocks are processed in parallel using the SYCL `parallel_for()`.

The DCT process converts image data from the pixel representation (where each pixel's color value is stored) to a sum of cosine representation. The color pattern of subsets of the image is represented as the sum of multiple cosine functions. In an **8x8** image, only eight discrete cosine functions are needed to produce the entire image. The only information needed to reconstruct the image is the coefficient associated with each cosine function. This is why the image is processed in **8x8** blocks. The DCT process converts an **8x8** matrix of pixels into an **8x8** matrix of these coefficients.

The quantizing process allows this data to be compressed to a smaller size than the original image. Each element of the matrix yielded by the DCT process is divided by a corresponding element of a quantizing matrix. This quantizing matrix is designed to reduce the number of coefficients required to represent the image by prioritizing the cosine functions most significant to the image's definition. The resulting matrix from this quantization step will look like a series of numbers followed by many zeros if read diagonally (which is how the data is stored in memory, allowing the large series of zeros to be compressed).

## Prerequisites
| Optimized for            | Description
|:---                      |:---
| OS                       | Ubuntu* 18.04 <br> Windows* 10
| Hardware                 | Skylake with GEN9 or newer
| Software                 | Intel&reg; oneAPI DPC++/C++ Compiler

## Key Implementation Details
The basic implementation explained in the code includes device selector, buffer, accessor, kernel, and command groups.

You can run the program in two modes using a preprocessor definition supplied at compile-time:

- With no preprocessor definition, the code will run the DCT process one time on the input image and write it to the output file.
- With `PERF_NUM` enabled, the code will process the input image five times, write it to the output, and print the average execution time to the console.

Because the image undergoes de-quantizing and DCT before being written to a file, the output image data will not be more compact than the input. However, it will reflect the image artifacts caused by lossy compression methods such as JPEG.

The `ProcessImage()` function uses a `parallel_for()` to calculate the index of each **8x8** block. It passes that index to the `ProcessBlock()` function, which performs the DCT and Quantization steps, along with de-quantization and IDCT.

The DCT representation is calculated through the multiplication of a DCT matrix (created by calling the `CreateDCT()` function) by a given color channel's data matrix. The resulting matrix is then multiplied by the inverse of the DCT matrix. The quantization calculation is performed by dividing each element of the resulting matrix by its corresponding element in the chosen quantization matrix. The inverse operations are performed to produce the de-quantized matrix and then the raw image data.

The program will attempt to run on a compatible GPU. If a compatible GPU is not found, the program will execute on the CPU (host device) instead. The program displays the device used in the output along with the time elapsed for rendering the image.

>**Note**: For comprehensive information about oneAPI programming, see the [Intel&reg; oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide). (Use search or the table of contents to find relevant information quickly.)

## Build the `Discrete Cosine Transform` Program for CPU and GPU

### Setting Environment Variables
When working with the Command Line Interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures your compiler, libraries, and tools are ready for development.

> **Note**: If you have not already done so, set up your CLI environment by sourcing the `setvars` script located in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: `. ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `$ bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\"Program Files (x86)"\Intel\oneAPI\setvars.bat`
> - For Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> Microsoft Visual Studio:
> - Open a command prompt window and execute `setx SETVARS_CONFIG " "`. This only needs to be set once and will automatically execute the `setvars` script every time Visual Studio is launched.
>
>For more information on environment variables, see "Use the setvars Script" for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

You can use [Modulefiles scripts](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-modulefiles-with-linux.html) to set up your development environment. The modulefiles scripts work with all Linux shells.

If you wish to fine tune the list of components and the version of those components, use
a [setvars config file](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos/use-a-config-file-for-setvars-sh-on-linux-or-macos.html) to set up your development environment.

### Use Visual Studio Code* (VS Code) (Optional)
You can use Visual Studio Code* (VS Code) extensions to set your environment, create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 1. Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 2. Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 3. Open a terminal in VS Code (**Terminal > New Terminal**).
 4. Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

### On Linux*
1. Change to the sample directory.
2. Build the program.
    ```
    $ mkdir build
    $ cd build
    $ cmake .. (or "cmake -D PERF_NUM=1 .." to enable performance test)
    $ make
    ```

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```

### On Windows*
**Using Visual Studio***

Build the program using **Visual Studio 2017** or newer.
1. Change to the sample directory.
2. Right-click on the solution file and open the solution in the IDE.
3. Enable performance metrics by selecting **Intel Release** for normal execution or **Intel Performance Test**.
4. Right-click on the project in **Solution Explorer** and select **Rebuild**.
5. From the top menu, select **Debug** > **Start without Debugging**.

**Using MSBuild**
1. Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019" or whatever is appropriate for your Visual Studio* version.
2. Change to the sample directory.
3. Run the following command: `MSBuild DCT.sln /t:Rebuild /p:Configuration="Intel Release"` (or `MSBuild DCT.sln /t:Rebuild /p:Configuration="Intel Performance Test"` for performance metrics. The program will process the image five times and provide the average processing time.)

#### Troubleshooting
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.

## Run the `Discrete Cosine Transform` Program
### Configurable Parameters
The program has two configurable parameters and requires an image input.

| Parameter                     | Description
|:---                           |:---
| Quantization levels           |Set these levels by changing which of the `quant[]` array definitions is used inside of `ProcessBlock()`. Uncomment the chosen quantization level and leave the others commented out.
| Queue definition              | `ProcessImage()` uses the SYCL default selector, which will prioritize offloading to GPU but will run on the host device (CPU) if no compatible GPU is found. You can force the code to run on the CPU by `changing default_selector{}` to `cpu_selector{}` on line 220.

You must specify an input bitmap (.bmp) image to process. The general usage syntax is as follows:
```
dct <input image file> <output inage file name>
```
where:
- `<input image file>` is the directory path and full .bmp image name of the file to process.
- `<output image file name` is the directory and full name to assign to the processed .bmp image file.

### On Linux
1. Run the program.
   ```
   make run
   ```
2. Clean the project files. (Optional)
   ```
   make clean
   ```

### On Windows
1. Change to the output directory.
2. Specify both the input image and the output file name, and run the program.
   ```
   DCT.exe ../../res/willyriver.bmp ../../res/willyriver_processed.bmp
   ```

## Example Output
```
Filename: willyriver.bmp W: 5184 H: 3456

Start image processing with offloading to GPU...
Running on Intel(R) UHD Graphics 620
--The processing time is 6.27823 seconds

DCT successfully completed on the device.
The processed image has been written to willyriver_processed.bmp
```

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
