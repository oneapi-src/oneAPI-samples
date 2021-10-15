# Getting Started Sample for Intel oneAPI Rendering Toolkit: Intel Open VKL

This sample program, vklTutorial, shows sampling output amongst the different volumetric sampling capabilities with Intel Open VKL.

Output is written to the console (stdout).

## License

TBD

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Requirements

To build and run the samples you will need a compiler toolchain and imaging tools:

Compiler:
- MSVS 2019 on Windows* OS
- On other platforms a C++11 compiler and a C99 compiler. (Ex: gcc/g++/clang)

oneAPI Libraries:
Install the Intel oneAPI Rendering Toolkit
- Embree
- Open VKL




## Build and Run

### Windows OS:

Run a new **x64 Native Tools Command Prompt for MSVS 2019**

```
call <path-to-oneapi-folder>\setvars.bat
cd <path-to-oneAPI-samples>\RenderingToolkit\openvkl_gsg
mkdir build
cd build
cmake -G"Visual Studio 16 2019" -A x64 -DCMAKE_PREFIX_PATH="<path-to-oneapi-folder>" ..
cmake --build . --config Release
cd Release
vklTutorial.exe
```

Review the terminal output (stdout)


### Linux OS:

Start a new Terminal session
```
source <path-to-oneapi-folder>/setvars.sh
cd <path-to-oneAPI-samples>/RenderingToolkit/openvkl_gsg
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="<path-to-oneapi-folder>" ..
cmake --build .
./vklTutorial
```

Review the terminal output (stdout)


### MacOS:

Start a new Terminal session

```
source <path-to-oneapi-folder>/setvars.sh
cd <path-to-oneAPI-samples>/RenderingToolkit/openvkl_gsg
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="<path-to-oneapi-folder>" ..
cmake --build .
./vklTutorial
```

Review the terminal output (stdout)
