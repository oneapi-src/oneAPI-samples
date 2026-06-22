# ospTutorialGLM
 This is a small example tutorial how to use OSPRay in an application using GLM instead of rkcommon for math types.
## Build and Run

### Windows

1. Run a new **x64 Native Tools Command Prompt for MSVS 2019**.

```
call <path-to-oneapi-folder>\setvars.bat
cd <path-to-oneAPI-samples>\RenderingToolkit\Tutorial\ospTutorialGLM
mkdir build
cd build
cmake ..
cmake --build . 
cd Debug
ospTutorialGLM.exe
```

### Linux

1. Start a new Terminal session.
```
source <path-to-oneapi-folder>/setvars.sh
cd <path-to-oneAPI-samples>/RenderingToolkit/Tutorial/ospTutorialGLM
mkdir build
cd build
cmake ..
cmake --build .
./ospTutorialGLM
```

### macOS

1. Start a new Terminal session.

```
source <path-to-oneapi-folder>/setvars.sh
cd <path-to-oneAPI-samples>/RenderingToolkit/Tutorial/ospTutorialGLM
mkdir build
cd build
cmake ..
cmake --build .
./ospTutorialGLM
```

### Additional Notes

oneAPI Rendering Toolkit 2023.1 version's cmake file contains an errata. The errata will produce an error while building the example. Please apply the following workaround described in the following page. 2023.1.1 version will address the issue.

https://community.intel.com/t5/Intel-oneAPI-Rendering-Toolkit/2023-1-troubleshoot-errata-CMake-Error/m-p/1476040#M98
