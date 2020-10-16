# `hello-vpp` Sample

This sample shows how to use the oneAPI Video Processing Library (oneVPL) to
perform simple video processing.

| Optimized for       | Description
|-------------------- | ----------------------------------------
| OS                  | Ubuntu* 18.04; Windows* 10
| Hardware            | Intel® Processor Graphics GEN9 or newer
| Software            | Intel® oneAPI Video Processing Library(oneVPL)
| What You Will Learn | How to use oneVPL to resize an I420 raw video file
| Time to Complete    | 5 minutes

## Purpose

This sample is a command line application that takes a file containing a raw
I420 format video elementary stream as an argument, processes it with oneVPL
writes the resized output to `out.i420` in I420 raw video format.


| Configuration     | Default setting
| ----------------- | ----------------------------------
| Target device     | CPU
| Input format      | I420
| Output format     | I420
| Output resolution | 640 x 480

## License

This code sample is licensed under MIT license.


## Building the `hello-vpp` Program

### On a Linux* System

Perform the following steps:

1. Install the prerequisite software. To build and run the sample you need to
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
   installation, which is which is `C:\Program Files (x86)\Intel\oneAPI\`
   when installed using default options. If you customized the installation
   folder, the `setvars.bat` is in your custom location.  Note that if a
   compiler is not part of your oneAPI installation you should run in a Visual
   Studio 64-bit command prompt.

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


#### Building the program using VS2017 or VS2019 IDE

1. Install the Intel® oneAPI Base Toolkit for Windows*
2. Right click on the solution file and open using either VS2017 or VS2019 IDE.
3. Right click on the project in Solution explorer and select Rebuild.
4. From top menu select Debug -> Start without Debugging.


## Running the Sample

### Application Parameters

The instructions given above run the sample executable with the argument
`<sample_dir>/content/cars_128x96.i420 128 96`.


### Example of Output

```
Processing hello-vpp/content/cars_128x96.i420 -> out.i420
Processed 60 frames

```

You can find the output file `out.i420` in the build directory and its size is `640x480`.

You can display the output with a video player that supports raw streams such as
FFplay. You can use the following command to display the output with FFplay:

```
ffplay -video_size 640x480 -pixel_format yuv420p -f rawvideo out.i420
```
