# `dpcpp-blur` Sample

This sample shows how to use a DPC++ kernel together with the oneAPI Video Processing Library(oneVPL) to perform a simple video content blur.

| Optimized for   | Description
|---------------- | ----------------------------------------
| OS              | Ubuntu* 20.04; Windows* 10
| Hardware        | CPU: See [System Requirements](https://software.intel.com/content/www/us/en/develop/articles/oneapi-video-processing-library-system-requirements.html)
|                 | GPU: Future Intel® Graphics platforms supporting oneVPL 2.x API features 
| Software        | Intel® oneAPI Video Processing Library (oneVPL)
| What You Will Learn | How to use oneVPL and DPC++ to convert a raw video file into BGRA color format and blur each frame.
| Time to Complete | 5 minutes

Native raw frame input format: CPU=I420, GPU=NV12.

## Purpose

This sample is a command line application that takes a file containing a raw frame input video as an argument. 
Using oneVPL, the application converts it to BGRA and blurs each frame with DPC++ using the SYCL kernel. 
The decoded output is then written to `out.raw`in BGRA format.

If the oneAPI DPC++ Compiler is not found, the blur operation will be disabled.


## Key Implementation details

| Configuration     | Default setting
| ----------------- | ----------------------------------
| Target device     | CPU
| Input format      | I420
| Output format     | BGRA raw video elementary stream
| Output resolution | 256x192


## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the `dpcpp-blur` Program

### Include Files
The oneVPL include folder is located at these locations on your development system:
 - Windows: %ONEAPI_ROOT%\vpl\latest\include 
 - Linux: $ONEAPI_ROOT/vpl/latest/include

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU) and whether to run in batch or interactive mode. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)


### On a Linux* System

Perform the following steps:

1. Install the prerequisite software. To build and run the sample, you need to
   install prerequisite software and set up your environment:

   - Intel® oneAPI Base Toolkit for Linux*
   - [CMake](https://cmake.org)

2. Set up your environment using the following command.
   ```
   source <oneapi_install_dir>/setvars.sh
   ```
   Here `<oneapi_install_dir>` represents the root folder of your oneAPI
   installation, which is `/opt/intel/oneapi/` when installed as root, and
   `~/intel/oneapi/` when installed as a normal user.  If you customized the
   installation folder, it is in your custom location.

3. Build the program using the following commands:
   ```
   mkdir build
   cd build
   cmake ..
   cmake --build .
   ```

4. Run the program using the following command:
   ```
   cmake --build . --target run
   ```


### On a Windows* System Using Visual Studio* Version 2017 or Newer

#### Building the program using CMake

1. Install the prerequisite software. To build and run the sample you need to
   install prerequisite software and set up your environment:

   - Intel® oneAPI Base Toolkit for Windows*
   - [CMake](https://cmake.org)

2. Set up your environment using the following command.
   ```
   <oneapi_install_dir>\setvars.bat
   ```
   Here `<oneapi_install_dir>` represents the root folder of your oneAPI
   installation, which is `C:\Program Files (x86)\Intel\oneAPI\`
   when installed using default options. If you customized the installation
   folder, the `setvars.bat` is in your custom location.  Note that if a
   compiler is not part of your oneAPI installation, you should run in a Visual
   Studio 64-bit command prompt.

3. Build the program using the following commands:
   ```
   mkdir build
   cd build
   cmake .. -T "Intel(R) oneAPI DPC++ Compiler"
   cmake --build . --config Release
   ```

4. Run the program using the following command:
   ```
   cmake --build . --target run --config Release
   ```


#### Building the program using VS2017 or VS2019 IDE

1. Install the Intel® oneAPI Base Toolkit for Windows*
2. Right-click on the solution file and open using either VS2017 or VS2019 IDE.
3. Right-click on the project in Solution Explorer and select Rebuild.
4. From the top menu, select Debug -> Start without Debugging.

***
Note: You need Base Toolkit 2021.2 or later to build this sample with the IDE.
***


## Running the Sample

### Application Parameters

The instructions given above run the sample executable with the argument
`-i <sample_dir>/content/cars_128x96.i420 -w 128 -h 96`.


### Example of Output

```
Processing ../content/cars_128x96.i420 -> out.raw
Processed 60 frames
```

You can find the output file ``out.raw`` in the build directory.

You can display the output with a video player that supports raw streams such as
FFplay. You can use the following command to display the output with FFplay:

```
ffplay -video_size 256x192 -pixel_format bgra -f rawvideo out.raw
```
