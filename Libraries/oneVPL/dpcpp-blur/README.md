# `dpcpp-blur` Sample

This sample shows how to use a DPC++ kernel together with
oneAPI Video Processing Library to perform a simple video content blur.

| Optimized for   | Description
|---------------- | ----------------------------------------
| OS              | Ubuntu* 18.04
| Hardware        | Intel® Processor Graphics GEN9 or newer
| Software        | Intel® oneAPI Video Processing Library (oneVPL)
| What You Will Learn | How to use oneVPL and DPC++ to convert I420 raw video files into BGRA and blur each frame.
| Time to Complete | 5 minutes

* I420: YUV color planes
* BGRA: BGRA color planes

## Purpose

This sample is a command line application that takes a file containing a raw
I420 format video elementary stream as an argument, converts it to BGRA with
oneVPL and blurs each frame with DPC++ by using SYCL kernel, and writes the
decoded output to `out.bgra` in BGRA format.

If the oneAPI DPC++ Compiler is not found, the blur operation will be disabled.



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
`<sample_dir>/content/cars_128x96.i420 128 96`.


### Example of Output

```
Processing dpcpp-blur/content/cars_128x96.i420 -> out.bgra
Processed 60 frames
```

You can find the output file ``out.bgra`` in the build directory.

You can display the output with a video player that supports raw streams such as
FFplay. You can use the following command to display the output with FFplay:

```
ffplay -video_size [128]x[96] -pixel_format bgra -f raw video out.bgra
```
