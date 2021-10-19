# `hello-decode` Sample

This sample shows how to use the oneAPI Video Processing Library (oneVPL) to
perform a simple video decode using 2.x enhanced programming model APIs.

| Optimized for    | Description
|----------------- | ----------------------------------------
| OS               | Ubuntu* 20.04
| Hardware         | Intel® Processor Graphics GEN9 or newer
| Software         | Intel® oneAPI Video Processing Library(oneVPL)
| What You Will Learn | How to use oneVPL to decode an H.265 encoded video file
| Time to Complete | 5 minutes


## Purpose

This sample is a command line application that takes a file containing an H.265
video elementary stream as an argument. Using oneVPL, the application decodes
and writes the decoded output to a file `out.raw` in raw format.

Native raw frame output format: CPU=I420, GPU=NV12.

## Key Implementation details

| Configuration     | Default setting
| ----------------- | ----------------------------------
| Target device     | CPU
| Input format      | H.265 video elementary stream
| Output format     | I420
| Output resolution | same as the input


## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)


## Building the `hello-decode` Program

### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

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
`-i examples/content/cars_128x96.h265`.


### Example of Output

```
Implementation info
      version = 2.4
      impl = Software
Decoding hello-decode/../content/cars_128x96.h265 -> out.raw
Decoded 60 frames
```

You can find the output file `out.raw` in the build directory.

You can display the output with a video player that supports raw streams such as
FFplay. You can use the following command to display the output with FFplay:

```
ffplay -video_size 128x96 -pixel_format yuv420p -f rawvideo out.raw
```

Use nv12 for pixel_format for GPU output.
