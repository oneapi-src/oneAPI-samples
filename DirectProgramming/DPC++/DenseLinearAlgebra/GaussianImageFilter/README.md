# Image Gaussian Blur example program

## Purpose
This SYCL code example implements a Gaussian blur filter, blurring
either a JPG or PNG image from the command line. The original file is not modified.
The output file is in a PNG format.

__Output Image:__

![Gaussian blur input](images/sample_image.jpg)<br>
![Gaussian blur output](images/sample_image-blurred.png)

## Prerequisites

| Minimum Requirements              | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 20.04.5 LTS
| Hardware                          | Intel&reg; 11th Gen Intel Core i7-1185G7 + Mesa Intel Xe Graphics
| Compiler Toolchain                | Visual Studio Code IDE, Intel oneAPI Base Toolkit (inc its prerequisite)
| Libraries                         | Install Intel oneAPI Base Toolkit 
| Tools                             | Visual Studio Code 1.73.1, VSCode Microsoft C/C++ extns, a .png capable image viewer

## Build and Run using Visual Code Studio

### Linux*

Within a terminal window change directory to this project's folder. At the 
terminal prompt type:

```
cd ImageGuassianBlur
code .
```

Visual Studio Code will open this project displaying its files in the Explorer 
pane. 
The project is already set up with build configurations to build either a
debug build or a release build of the program. When a program is built, it is
placed in the bin directory of this project's top folder. 

To build the program hit Ctrl+Shift+b and choose the type of program to build.
The debug executable will have a '_d' appended to its name.

To blur an image, copy the images/sample_image.jpg to the bin directory.
To execute the program, type in the Visual Studio Code terminal window:
```
cd bin
./gaussian_blur_d sample_image.jpg
```
A new image file will appear in the bin directory 'sample_image-blurred.png'.
To view the image, select it in the directory folder app and hit return.
Ubuntu will display the image using the preview app.

## Build and Run using CMake
### Linux*
```
mkdir build
cd build
cmake ..
make
```

To blur an image, copy the images/sample_image.jpg to the directory of the new
executable. Type in the terminal window:

```
cd build/src
./gaussian_blur sample_image.jpg
```
Open the resulting file: `sample_image-blurred.png` with an image viewer.

## Debug the program using Visual Studio Code

### Linux*

Due to an issue with the image load library function stbi_load, make the 
directory bin (if it does not exist already) and copy the sample_image.jpg
file into it. This will allow the program to find the file and continue the
debug session.

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
