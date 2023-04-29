# ospTutorialGLM

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
