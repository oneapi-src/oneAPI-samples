# `dpcpp-blur` Sample

This sample shows how to use a DPC++ kernel together with
oneAPI Video Processing Library to perform a simple video content blur.

| Optimized for    | Description
|----------------- | ----------------------------------------
| OS               | Ubuntu* 20.04
| Hardware         | Intel® Processor Graphics GEN9 or newer
| Software         | Intel® oneAPI Video Processing Library(oneVPL)
| What You Will Learn | How to use oneVPL and DPC++ to convert raw video files into BGRA and blur each frame.
| Time to Complete | 5 minutes

Expected input/output formats:
* In: CPU=I420 (yuv420p color planes), GPU=NV12 color planes
* Out: BGRA color planes

## Purpose

This sample is a command line application that takes a file containing a raw
format video file as an argument, converts it to BGRA with oneVPL, blurs each frame with DPC++ by using SYCL kernel,
and writes the processed output to `out.bgra` in BGRA format.

GPU optimization is available in Linux, including oneAPI level zero optimizations allowing the kernel to run 
directly on VPL output without copies to/from CPU memory.

## Key Implementation details

| Configuration     | Default setting
| ----------------- | ----------------------------------
| Target device     | CPU
| Input format      | I420
| Output format     | BGRA raw video elementary stream
| Output resolution | same as input


## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the `dpcpp-blur` Program

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

   Additional setup steps to enable GPU execution can be found here:
   https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-focal.html

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

The instructions given above run the sample executable with these arguments
`-i <sample_dir>/content/cars_128x96.i420 -w 128 -h 96`.

In Linux, an additional '-hw' parameter will run on GPU if GPU stack components 
are found in your environment.

### Example of Output

```
Queue initialized on Intel(R) Core(TM) i5-9400 CPU @ 2.90GHz
Implementation details:
  ApiVersion:           2.4  
  Implementation type:  SW
  AccelerationMode via: NA 
  Path: /opt/intel/oneapi/vpl/2021.4.0/lib/libvplswref64.so.1

Processing ../../content/cars_128x96.nv12 -> out.raw
Processed 60 frames

```

You can find the output file ``out.raw`` in the build directory.

You can display the output with a video player that supports raw streams such as
FFplay. You can use the following command to display the output with FFplay:

```
ffplay -video_size 256x192 -pixel_format bgra -f raw video out.bgra
```
