# `DPC++ Discrete Cosine Transform` Sample

Discrete Cosine Transform (DCT) and Quantization are the first two steps in the JPEG compression standard. This sample demonstrates how DCT and Quantizing stages can be implemented to run faster using Data Parallel C++ (DPC++) by offloading the work of image processing to a GPU or other device.

For comprehensive instructions regarding DPC++ Programming, go to https://software.intel.com/en-us/oneapi-programming-guide and search based on relevant terms noted in the comments.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04; Windows 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel® oneAPI DPC++/C++ Compiler;
| What you will learn               | How to parallel process image data using DPC++ for producing a Discrete Cosine Transform
| Time to complete                  | 15 minutes


## Purpose

DCT is a lossy compression algorithm that is used to represent every data point value using the sum of cosine functions, which are linearly orthogonal to each other. The program shows the possible effect of quality reduction in the image when one performs DCT, followed by quantization as found in JPEG compression.

This program generates an output image by first processing an input image using DCT and quantization, undoing the process through Inverse DCT and de-quantizing, and then writing the output to a BMP image file. The processing stage is performed on 8x8 subsections of the pixel image, referred to as 'blocks' in the code sample.

Since individual blocks of data can be processed independently, the overall image can be decomposed and processed in parallel, where each block represents a work item. Using DPC++, parallelization can be implemented relatively quickly, and with few changes to a serial version. The blocks are processed in parallel using the SYCL parallel_for(), and the code will attempt to first execute on an available GPU and fallback to the system's CPU if a compatible GPU is not detected. The device used for the compilation is displayed in the output along with elapsed time to render the processed image.

The DCT process converts image data from the pixel representation (where the color value of each pixel is stored) to a sum of cosine representation, where the color pattern of subsets of the image is represented as the sum of multiple cosine functions. In an 8x8 image, only eight discrete cosine functions are needed to produce the entire image, and the only information needed to reconstruct the image is the coefficient associated with each cosine function. This is why the image is processed in 8x8 blocks. The DCT process converts an 8x8 matrix of pixels into an 8x8 matrix of these coefficients.

The quantizing process is what allows this data to be compressed to a smaller size than the original image. Each element of the matrix yielded by the DCT process is divided by a corresponding element of a quantizing matrix. This quantizing matrix is designed to reduce the number of coefficients required to represent the image, by prioritizing the cosine functions which are most significant to the image's definition. The resulting matrix from this quantization step will look like a series of numbers followed by many zeros if read diagonally (which is how the data is stored in memory, allowing the large series of zeros to be compressed).

The Code Sample can be run in two different modes, based on preprocessor definitions supplied at compile-time: 

* With no preprocessor definition, the code will run the DCT process one time on the input image and write it to the output file. 
* With PERF_NUM enabled, the code will process the input image five times, write it to the output, and print the average execution time to the console.

Because the image undergoes de-quantizing and IDCT before being written to a file, the output image data will not be more compact than the input. However, it will reflect the image artifacts caused by lossy compression methods such as JPEG.


## Key Implementation Details 

The basic DPC++ implementation explained in the code includes device selector, buffer, accessor, kernel, and command groups.

The ProcessImage() function uses a parallel_for() to calculate the index of each 8x8 block. It passes that index to the ProcessBlock() function, which performs the DCT and Quantization steps, along with de-quantization and IDCT. 

The DCT representation is calculated through the multiplication of a DCT matrix (created by calling the CreateDCT() function) by a given color channel's data matrix, with the resulting matrix then multiplied by the inverse of the DCT matrix. The quantization calculation is performed through the division of each element of the resulting matrix by its corresponding element in the chosen quantization matrix. The inverse operations are performed to produce the de-quantized matrix and then the raw image data.

 
## License  

This code sample is licensed under MIT license. 


## Building the `DPC++ Discrete Cosine Transform` Program for CPU and GPU

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) as well whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Linux* System
Perform the following steps:
1. Build the program with `cmake` using the following shell commands.
    From the root directory of the DCT project:
    ``` 
    $ mkdir build
    $ cd build
    $ cmake .. (or "cmake -D PERF_NUM=1 .." to enable performance test)
    $ make
    ```

2. Run the program:
    ```
    make run
    ```

3. Clean the program using:
    ```
    make clean
    ```

### On a Windows* System Using Visual Studio* Version 2017 or Newer
* Build the program using VS2017 or VS2019
      Right-click on the solution file and open using either VS2017 or VS2019 IDE.
      Set the configuration to 'Intel Release' for normal execution or 'Intel Performance Test' to take performance metrics.
      Right-click on the project in Solution Explorer and select Rebuild.

      To run:
      From the top menu, select Debug -> Start without Debugging.

* Build the program using MSBuild
      Open "Intel oneAPI command prompt for Microsoft Visual Studio 2019" and use your shell of choice to navigate to the DCT sample directory
      Run command - MSBuild DCT.sln /t:Rebuild /p:Configuration="Intel Release"     (or Configuration="Intel Performance Test" for performance tabulation)

      To run:
      Run command - '.\x64\Intel Release\DCT.exe' ./res/willyriver.bmp ./res/willyriver_processed.bmp


## Running the Sample

### Application Parameters 
Different levels of quantization can be set by changing which of the quant[] array definitions is used inside of ProcessBlock(). Uncomment the chosen quantization level and leave the others commented out.

The queue definition in ProcessImage() uses the SYCL default selector, which will prioritize offloading to GPU but will run on the host device if none is found. You can force the code to run on the CPU by changing default_selector{} to cpu_selector{} on line 220.

### Example of Output
```
Filename: ..\res\willyriver.bmp W: 5184 H: 3456

Start image processing with offloading to GPU...
Running on Intel(R) UHD Graphics 620
--The processing time is 6.27823 seconds

DCT successfully completed on the device.
The processed image has been written to ..\res\willyriver_processed.bmp
```