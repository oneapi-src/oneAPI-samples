# Monte Carlo Pi example program

## Purpose
Monte Carlo Simulation is a broad category of computation that utilizes
statistical analysis to reach a result. This  `Monte Carlo Pi` sample uses the
Monte Carlo Procedure to estimate the value of pi.

## Prerequisites

| Minimum Requirements              | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 20.04.5 LTS
| Hardware                          | Intel&reg; 11th Gen Intel Core i7-1185G7 + Mesa Intel Xe Graphics
| Compiler Toolchain                | Visual Studio Code IDE, Intel oneAPI Base Toolkit (inc its prerequisite)
| Libraries                         | Install Intel oneAPI Base Toolkit 
| Tools                             | Visual Studio Code 1.73.1, VSCode Microsoft C/C++ extns

## Build and Run using Visual Code Studio

### Linux*

Within a terminal window change directory to this project's folder. At the 
terminal prompt type:

```
cd MonteCarloPi_v2
code .
```

Visual Studio Code will open this project displaying its files in the Explorer 
pane. 
The project is already set up with build configurations to build either a
debug build or a release build of the program. When a program is built, it is
placed in the bin directory of this project's top folder. 

To build the program hit Ctrl+Shift+b and choose the type of program to build.
The debug executable will have a '_d' appended to its name.

To execute the program, type in the Visual Studio Code terminal window:
```
cd bin
./MonteCarloPi_d cpu
```

## Build and Run using CMake
### Linux*
```
mkdir build
cd build
cmake ..
make
```

To execute the program type in the terminal window:

```
cd build/src
./MonteCarloPi cpu
```

## Debug the program using Visual Studio Code

### Linux*

To debug the program, either choose from the IDE's run menu 
'Start debugging' or hit F5 on the keyboard.
The debug launch.json configuration file defines the debug session to:
* To halt the program at the first line of code after main().
Use the GUI debug panel's buttons to step over code (key F10) lines to see the 
program advance. 
Breakpoints can be set either in the main code or the kernel code.

Note: Setting breakpoints in the kernel code does not present the normal 
      step through code behavior. Instead a breakpoint event is occurring
      on each thread being executed and so switches to the context of 
      that thread. To step through the code of a single thread, use the 
      Intel gdb-oneapi command 'set scheduler-locking step' or 'on' in the 
      IDE's debug console prompt. As this is not the main thread, be sure
      to revert this setting on returning to debug any host side code. 
      Use the command 'set scheduler-locking replay' or 'off'.  

## License

Code samples are licensed under the Apache 2.0 license. See
[LICENSE.txt](LICENSE.txt) for details.
