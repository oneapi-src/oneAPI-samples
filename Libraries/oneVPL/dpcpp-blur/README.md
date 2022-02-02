# `dpcpp-blur` Sample

This sample shows how to use a DPC++ kernel together with
oneAPI Video Processing Library to perform a simple video content blur.

| Optimized for    | Description
|----------------- | ----------------------------------------
| OS               | Ubuntu* 20.04
| Hardware         | CPU: See [System Requirements](https://software.intel.com/content/www/us/en/develop/articles/oneapi-video-processing-library-system-requirements.html)
|                  | GPU: Compatible with Intel® oneAPI Video Processing Library(oneVPL) GPU implementation, which can be found at https://github.com/oneapi-src/oneVPL-intel-gpu 
| Software         | oneAPI Video Processing Library (oneVPL)
| What You Will Learn | How to use oneVPL and DPC++ to convert raw video files into BGRA and blur each frame.
| Time to Complete | 5 minutes

Expected input/output formats:
* In: CPU=I420 (yuv420p color planes), GPU=NV12 color planes
* Out: BGRA color planes

## Purpose

This sample is a command line application that takes a file containing a raw
format video file as an argument, converts it to BGRA with oneVPL, blurs each frame with DPC++ by using SYCL kernel,
and writes the processed output to `out.bgra` in BGRA format.

GPU optimization is available in Linux, including oneAPI Level Zero optimizations allowing the kernel to run
directly on VPL output without copies to/from CPU memory.

## Key Implementation details

| Configuration     | Default setting
| ----------------- | ----------------------------------
| Target device     | CPU
| Input format      | I420
| Output format     | BGRA raw video elementary stream
| Output resolution | 256 x 192


## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.


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
   cp $ONEAPI_ROOT/vpl/latest/examples .
   cd examples/interop/dpcpp-blur
   mkdir build
   cd build
   cmake ..
   cmake --build .
   ```

4. Run the program with default arguments using the following command:
   ```
   cmake --build . --target run
   ```

## Running the Sample

### Application Parameters

The instructions given above run the sample executable with these arguments
`-i ${CONTENTPATH}/cars_128x96.i420 -w 128 -h 96`.

In Linux, an additional '-hw' parameter will run on GPU if GPU stack components
are found in your environment.

### Example Output

```
Queue initialized on 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz
Implementation details:

  ApiVersion:           2.5  
  Implementation type:  SW
  AccelerationMode via: NA 
  DeviceID:             0000 
  Path: /opt/intel/oneapi/vpl/2021.6.0/lib/libvplswref64.so.1

Processing /home/test/intel_innersource/frameworks.media.onevpl.dispatcher/examples/interop/dpcpp-blur/content/cars_128x96.i420 -> out.raw
Processed 60 frames
```

You can find the 256x192 BGRA output file ``out.raw`` in the build directory.

You can display the output with a video player that supports raw streams such as
FFplay. You can use the following command to display the output with FFplay:

```
ffplay -video_size 256x192 -pixel_format bgra -f raw video out.bgra
```
