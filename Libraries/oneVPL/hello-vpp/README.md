# `hello-vpp` Sample

This sample shows how to use the oneAPI Video Processing Library (oneVPL) to
perform simple video processing using 2.x enhanced programming model APIs.

| Optimized for       | Description
|-------------------- | ----------------------------------------
| OS                  | Ubuntu* 20.04
| Hardware            | CPU: See [System Requirements](https://software.intel.com/content/www/us/en/develop/articles/oneapi-video-processing-library-system-requirements.html)
|                     | GPU: Future Intel® Graphics platforms supporting oneVPL 2.x API features 
| Software            | Intel® oneAPI Video Processing Library(oneVPL)
| What You Will Learn | How to use oneVPL to resize and change color format of a raw video file
| Time to Complete    | 5 minutes

## Purpose

This sample is a command line application that takes a file containing a raw
format video elementary stream as an argument. Using oneVPL, the application
processes it and writes the resized output to `out.raw` in BGRA raw video format.

Native raw frame input format: CPU=I420, GPU=NV12.

| Configuration     | Default setting
| ----------------- | ----------------------------------
| Target device     | CPU
| Input format      | I420
| Output format     | BGRA raw video elementary stream
| Output resolution | 640 x 480

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)


## Building the `hello-vpp` Program

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
## Running the Sample

### Application Parameters

The instructions given above run the sample executable with the argument
`-i examples/content/cars_128x96.i420 128 96`.


### Example of Output

```
Implementation info
      version = 2.4
      impl = Software
Processing hello-vpp/../content/cars_128x96.i420 -> out.raw
Processed 60 frames

```

You can find the output file `out.raw` in the build directory, and its size is `640x480`.

You can display the output with a video player that supports raw streams such as
FFplay. You can use the following command to display the output with FFplay:

```
ffplay -video_size 640x480 -pixel_format bgra -f rawvideo out.raw
```
