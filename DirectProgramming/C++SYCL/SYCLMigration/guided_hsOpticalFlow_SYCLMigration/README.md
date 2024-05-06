# `HSOpticalFlow` Sample

The `HSOpticalFlow` sample is a computation of per-pixel motion estimation between two consecutive image frames caused by the movement of an object or camera. The original CUDA* source code is migrated to SYCL for portability across GPUs from multiple vendors.

| Area                      | Description
|:---                       |:---
| What you will learn       | Migrate and optimize the HSOptical sample from CUDA to SYCL.
| Time to complete          | 15 minutes
| Category                  | Concepts and Functionality

> **Note**: This sample is based on the [HSOpticalFlow](https://github.com/NVIDIA/cuda-samples/tree/v11.8/Samples/5_Domain_Specific/HSOpticalFlow) sample in the NVIDIA/cuda-samples GitHub repository.
>
>For more information on Optical Flow Algorithm and CUDA implementation, refer to [*Optical Flow Estimation with Cuda*](https://github.com/NVIDIA/cuda-samples/blob/v11.8/Samples/5_Domain_Specific/HSOpticalFlow/doc/OpticalFlow.pdf) by Mikhail Smirnov.

## Purpose

The optical flow method is based on two assumptions: brightness constancy and spatial flow smoothness. These assumptions are combined in a single energy functional and a solution is found as its minimum point. The sample includes both parallel and serial computation, which allows direct results comparison between CPU and Device. Input images of the sample are computed to get the absolute difference value output(L1 error) between serial and parallel computation. The parallel implementation demonstrates the use of key SYCL concepts, such as

- Image Processing
- SYCL Image memory
- Sub-group primitives
- Shared Memory

This sample illustrates the steps needed for manual migration of CUDA Texture memory objects and APIs such as  `cudaResourceDesc`, `cudaTextureDesc`, and `cudaCreateTextureObject` to SYCL equivalent. These CUDA Texture memory APIs are manually migrated to SYCL Image memory APIs.

> **Note**: The sample used the open-source SYCLomatic tool that assists developers in porting CUDA code to SYCL code. To finish the process, you must complete the rest of the coding manually and then tune to the desired level of performance for the target architecture. You can also use the Intel® DPC++ Compatibility Tool available to augment the Base Toolkit.

This sample contains three versions in the following folders:

| Folder Name                  | Description
|:---                          |:---
| `01_dpct_output`             | Contains the output of the SYCLomatic tool used to migrate SYCL-compliant code from CUDA code. This SYCL code has some code that is not migrated and has to be manually fixed to get full functionality. (The code does not functionally work as supplied.)
| `02_sycl_migrated_optimized` | Contains manually migrated SYCL code from CUDA code with performance optimizations applied.

## Prerequisites

| Optimized for              | Description
|:---                        |:---
| OS                         | Ubuntu* 22.04
| Hardware                   | Intel® Gen9 <br> Intel® Gen11 <br> Intel® Xeon CPU <br> Nvidia Tesla P100 <br> Nvidia A100 <br> Nvidia H100
| Software                   | SYCLomatic (Tag - 20240403) <br> Intel® oneAPI Base Toolkit (Base Kit) version 2024.1 <br> oneAPI for NVIDIA GPUs plugin (version 2024.1) from Codeplay

For more information on how to install Syclomatic Tool, visit [Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html#gs.v354cy) <br>
Refer [oneAPI for NVIDIA GPUs plugin](https://developer.codeplay.com/products/oneapi/nvidia/) from Codeplay to execute the sample on NVIDIA GPU.

## Key Implementation Details

This sample demonstrates the migration of the following prominent CUDA features:

- CUDA Texture Memory API
- Shared memory
- Cooperative groups

HSOptical flow mainly involves the following stages image downscaling and upscaling, image warping, computing derivatives, and computation of Jacobi iteration.

Image scaling downscaling or upscaling aims to preserve the visual appearance of the original image when it is resized, without changing the amount of data in that image. An image with a resolution of width × height will be resized to new_width × new_height with a scale factor. A scale factor less than 1 indicates shrinking while a scale factor greater than 1 indicates stretching.

Image warping is a transformation that maps all positions in the source image plane to positions in a destination plane. Texture addressing mode is set to Clamp, and texture coordinates are unnormalized. Clamp addressing mode to handle the out-of-range coordinates. It eases computing derivatives and warping whenever we need to reflect out-of-range coordinates across borders.

Once the warped image is created, derivatives are computed. For each pixel, the required stencil points from the texture are fetched and convolved them with the filter kernel. In terms of CUDA, we can create a thread for each pixel. This thread fetches required data and computes the derivative.

The next step involves solving for Jacobi iterations. Border conditions are explicitly handled within the kernel. The number of iterations is fixed during computations. This eliminates the need for checking errors on every iteration. The required number of iterations can be determined experimentally. To perform one iteration of the Jacobi method at a particular point, we need to know the results of the previous iteration for its four neighbors. If we simply load these values from global memory each value will be loaded four times. We store these values in shared memory. This approach reduces the number of global memory accesses, provides better coalescing, and improves overall performance.

Prolongation is performed with bilinear interpolation followed by scaling. and are handled independently. For each output pixel, there is a thread that fetches the output value from the texture and scales it.

In CUDA texture memory is used to read and update image data and the equivalent in SYCL is image memory where image objects represent a region of memory managed by the SYCL runtime. The data layout of the image memory is deliberately unspecified to allow implementations to provide a layout optimal to a given device. When accessed on the host, image memory may be stored on temporary host memory. When accessed on a device, image data is stored in the device image memory, which can often be texture memory if the device supports it. In the case of Intel integrated graphics, there is no dedicated texture memory, so the L3 cache is utilized.

>**Note**: Refer to [Workflow for a CUDA* to SYCL* Migration](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/cuda-sycl-migration-workflow.html#gs.s2njvh) for general information about the migration workflow.

### CUDA Source Code Evaluation

The HSOptical Flow sample includes both serial and parallel implementation of the algorithm in flowGold.cpp and flowCUDA.cu files respectively. In the parallel implementation, the computation is distributed among the following six kernels:

- `AddKernel()` - Performs vector addition
- `ComputeDerivativesKernel()` - Computes temporal and spatial derivatives of images
- `DownscaleKernel()` - Computes image downsizing
- `JacobiIteration()` - Computes for Jacobi iteration with border conditions explicitly handled within the kernels 
- `UpscaleKernel()` - Upscales one component of an image displacement field
- `WarpingKernel()` - Warps image with given displacement field

The host code of downscale, Compute derivatives, Upscale, and Warping uses texture memory for image data computation. The final computed result of serial and parallel implementation is then compared based on the threshold value.

>**Note**: For information on how to use SYCLomatic, refer to the materials at *[Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html)*.


## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script each time you open a new terminal window. This practice ensures that compilers, libraries, and tools are ready for development.

## Migrate the `HSOpticalFlow` Code

### Migrate the Code using SYCLomatic

For this sample, the SYCLomatic Tool automatically migrates ~80% of the CUDA runtime APIs to SYCL. Follow these steps to generate the SYCL code using the compatibility tool.

1. Clone the required GitHub repository to your local environment.
   ```
   git clone https://github.com/NVIDIA/cuda-samples.git
   ```
2. Change to the HSOpticalFlow sample directory.
   ```
   cd cuda-samples/Samples/5_Domain_Specific/HSOpticalFlow/
   ```
3. Generate a compilation database with intercept-build
   ```
   intercept-build make
   ```
   The above step creates a JSON file named compile_commands.json with all the compiler invocations and stores the names of the input files and the compiler options.

4. Pass the JSON file as input to the SYCLomatic Tool. The result is written to a folder named dpct_output. The `--in-root` specifies the path to the root of the source tree to be migrated. The `--gen-helper-function` option will make a copy of the dpct header files/functions used in the migrated code into the dpct_output folder as `include` folder.
   ```
   c2s -p compile_commands.json --in-root ../../.. --gen-helper-function
   ```
   
### Manual Workarounds

The following warnings in the "DPCT1XXX" format are generated by the tool to indicate the code has not migrated by the tool and needs to be manually modified in order to complete the migration.

1. DPCT1059: SYCL only supports 4-channel image format. Adjust the code.
    ```
    texRes.res.pitch2D.desc = cudaCreateChannelDesc<float>();
    ```
    CUDA HSOptical Flow sample uses single channel image format and SYCL supports only 4 channel image formats. We must adjust two properties of image data manually.
    1. Image data type format - The data type of the image accessor should be `sycl::float4`.
    2. Image input layout - Image data should be padded for additional image channels.

    The following code block illustrates the image data type format and padding of image data for additional image channels.
    ```
    float *src_p = (float *)sycl::malloc_shared(height * stride * sizeof(sycl::float4), q);
    for (int i = 0; i < 4 * height * stride; i++) src_p[i] = 0.f;

    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        int index = i * stride + j;
        src_p[index * 4 + 0] = src_h[index];
        src_p[index * 4 + 1] = src_p[index * 4 + 2] = src_p[index * 4 + 3] = 0.f;}
    }
    ``` 


2. DPCT1007: Migration of cudaTextureDesc::readMode is not supported.
    ```
    texDescr.readMode = cudaReadModeElementType;
    ```
    In CUDA the read mode can be set through `cudaReadModeNormalizedFloat` or `cudaReadModeElementType`. If it is cudaReadModeNormalizedFloat the value returned by the texture fetch is floating-point type, if it is cudaReadModeElementType, no conversion is performed.
    In SYCL, By default sampler API read mode is set to element-by-element which is guided by the buffer, hence there is no need for mapping or workaround.


3. Along with these changes, we have also converted the Image memory APIs from dpct namespace to sycl namespace and manually mapped the SYCL APIs.
    The following code block is the CUDA Texture memory object creation and object API, the `texRes` object sets the CUDA resource descriptor variables, and `texDescr` object sets the CUDA texture descriptor variables.
    ```
     cudaTextureObject_t texFine;
     cudaResourceDesc texRes;

     texRes.resType = cudaResourceTypePitch2D;
     texRes.res.pitch2D.devPtr = (void *)src;
     texRes.res.pitch2D.desc = cudaCreateChannelDesc<float>();
     texRes.res.pitch2D.width = width;
     texRes.res.pitch2D.height = height;
     texRes.res.pitch2D.pitchInBytes = stride * sizeof(float);

     cudaTextureDesc texDescr;

     texDescr.normalizedCoords = true;
     texDescr.filterMode = cudaFilterModeLinear;
     texDescr.addressMode[0] = cudaAddressModeMirror;
     texDescr.addressMode[1] = cudaAddressModeMirror;
     texDescr.readMode = cudaReadModeElementType;

     cudaCreateTextureObject(&texFine, &texRes, &texDescr, NULL);
    ```
    
    The following SYCL code is the equivalent SYCL image APIs.
    
    In SYCL implementation, `sycl::image` defines a shared image data. Images can be 1-, 2-, and 3-dimensional, which are accessed using the accessor class. SYCL images are created from a host pointer, like buffers, and construct an image with the specified image channel_order and channel_type, range, and pitch, with a raw host pointer to the image data. On object destruction, the data will be copied to the specified host pointer unless a final pointer is specified using `set_final_data()` in which case that specified pointer will be used.

    ```
    auto texFine = sycl::image<2>(src_p, sycl::image_channel_order::rgba,
                                    sycl::image_channel_type::fp32,
                                    sycl::range<2>(width, height),
                                    sycl::range<1>(stride * sizeof(sycl::float4)));

    auto texDescr = sycl::sampler(
      sycl::coordinate_normalization_mode::unnormalized,
      sycl::addressing_mode::clamp_to_edge, sycl::filtering_mode::nearest);
  
    q.submit([&](sycl::handler &cgh) {
        auto tex_acc =
           texFine.template get_access<sycl::float4,
                                       sycl::access::mode::read>(cgh);

        cgh.parallel_for(
            ...
        );
    });
    ```
    The texture descriptor in SYCL is created using the image_sampler struct, which supports a slightly different configuration for sampling an image. SYCL images do not support normalized readMode. By default, Sampler API supports read mode as element by element, which is guided by the buffer. Consequently, we change the image coordinates to “unnormalized”. For the members of this struct, the addressing mode is set as clamp to edge, whereas the addressing mode in CUDA is set to Mirror. Since mirrored address mode is only supported by normalized coordinates and SYCL uses unnormalized coordinated the address mode set is clamp to edge. Mirror address mode mirrors the out-of-range texture coordinates at every integer boundary, and clamp to edge addressing mode clamps out-of-range image coordinates to the extent.

    In CUDA, texture coordinates are normalized. This behavior causes the coordinates to be specified in the floating point range [0.0, 1.0-1/N], and the filtering mode is set to linear filtering to support the textures that are configured to return floating-point data. But in SYCL the coordinates are unnormalized and the filtering mode is set to linear or nearest based on the type of the image data. The downscaleKernel and derivativeKernel use integer coordinates and hence filtering mode is set to nearest. The upscaleKernel and warpingKernel use floating-point coordinates and hence filtering mode is set to linear. The nearest filtering mode chooses the color of the nearest pixel. Linear filtering mode performs a linear sampling of adjacent pixels.

    To access an image `get_access()` accessor member function is used. The accessor element type specifies how the image should be read from or written to. It can be either int4, uint4, or float4.


4. As the image coordinates in SYCL image_sampler are set to unnormalized, we need to modify the texture fetch in the gold  implementation, i.e., Tex2D and Tex2Di functions in order to account for the unnormalized texture coordinates in SYCL.

    In the original code, the out-of-range texture coordinates are mirrored as shown below. These host texture fetch functions read from arbitrary positions within an image using bilinear interpolation. Note that, mirrored addressing modes for texture are supported only for normalized texture coordinates.
    ```
    if (ix0 < 0) ix0 = abs(ix0 + 1);
    if (iy0 < 0) iy0 = abs(iy0 + 1);

    if (ix0 >= w) ix0 = w * 2 - ix0 - 1;
    if (iy0 >= h) iy0 = h * 2 - iy0 - 1;
    ```

    In the modified gold implementation, the out-of-range coordinates are changed to be clamped to edge as shown below to account for unnormalized coordinates in SYCL.
    ```
    if (ix0 < 0) ix0 = 0;
    if (iy0 < 0) iy0 = 0;

    if (ix0 >= w) ix0 = w - 1;
    if (iy0 >= h) iy0 = h - 1;
    ```

> **Note**: You can find more information about image samplers and options for struct members in section *4.7.8. Image samplers* of the [SYCL™ 2020 Specification](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#subsec:samplers).

5. CUDA code includes a custom API `findCUDADevice` in helper_cuda file to find the best CUDA Device available.

```
    findCudaDevice (argc, (const char **) argv);
```
Since its a custom API SYCLomatic tool will not act on it and we can either remove it or replace it with the `sycl get_device()` API


### Optimizations

Once you migrate the CUDA code to SYCL successfully and you have functional code, you can optimize the code by using profiling tools, which can help in identifying the hotspots such as operations/instructions taking longer time to execute, memory utilization, and the like.

#### Memory Operation Optimization

Since the CUDA HSOptical Flow sample uses single channel image and SYCL supports only 4 channel image format we have manually adjusted two properties of image data. Image data type format and Image input layout where Image data are padded for additional image channels. For this purpose, we can create additional host and USM memory using `malloc_shared` that is padded as shown below.

```
int dataSize = height * stride * sizeof(float);
float *src_h = (float *)malloc(dataSize);
q.memcpy(src_h, src, dataSize).wait();

float *src_p =
    (float *)sycl::malloc_shared(height * stride * sizeof(sycl::float4), q);
for (int i = 0; i < 4 * height * stride; i++) src_p[i] = 0.f;

for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int index = i * stride + j;
      src_p[index * 4 + 0] = src_h[index];
      src_p[index * 4 + 1] = src_p[index * 4 + 2] = src_p[index * 4 + 3] = 0.f;}
}
```

Even though malloc_shared gives us shared memory allocation that is accessible on the host and on sycl Device it increases execution time as it creates a lot of unnecessary memory movement in code. To avoid unnecessary memory copies, we can replace the `malloc_shared` with `malloc_device`.

```
int dataSize = height * stride * sizeof(float);
float *pI0_h = (float *)sycl::malloc_host(height * stride * sizeof(sycl::float4), q);
float *I0_h = (float *)sycl::malloc_host(dataSize, q);

q.memcpy(I0_h, src, dataSize).wait();

for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int index = i * stride + j;
      pI0_h[index * 4 + 0] = I0_h[index];
      pI0_h[index * 4 + 1] = pI0_h[index * 4 + 2] = pI0_h[index * 4 + 3] = 0.f;}
}

q.memcpy(src_p, pI0_h, height * width * sizeof(sycl::float4)).wait();
```

malloc_device returns a pointer to the newly allocated memory on the specified device on success. This memory is not accessible on the host. Hence, we need to copy memory to the host when required. Also copying from malloc_host to malloc_device is faster than compared to C malloc to malloc_device.


## Build and Run the `HSOpticalFlow` Sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files (x86)\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)*


### On Linux*

1. Change to the sample directory.
2. Build the program.
   ```
   $ mkdir build
   $ cd build
   $ cmake .. or ( cmake -D NVIDIA_GPU=1 .. )
   $ make
   ```
    **Note**: By default, no flag are enabled during build which supports Intel® UHD Graphics, Intel® Gen9, Gen11, Xeon CPU. <br>
    Enable `NVIDIA_GPU` flag during build which supports NVIDIA GPUs.([oneAPI for NVIDIA GPUs](https://developer.codeplay.com/products/oneapi/nvidia/) plugin   from Codeplay is required to build for NVIDIA GPUs ) <br>

   By default, this command sequence will build the `02_sycl_migrated_optimized` version of the program.

3. Run the program.

   Run `02_sycl_migrated_optimized` on GPU.
   ```
   $ make run
   ```
   Run `02_sycl_migrated_optimized` for CPU.
    ```
    $ export ONEAPI_DEVICE_SELECTOR=opencl:cpu
    $ make run
    $ unset ONEAPI_DEVICE_SELECTOR
    ```

#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
$ make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/docs/oneapi/user-guide-diagnostic-utility/2024-1/overview.html) for more information on using the utility.
  
## License
Code samples are licensed under the MIT license. See
[License.txt](https://www.intel.com/content/www/us/en/docs/oneapi/user-guide-diagnostic-utility/2024-0/overview.html) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
