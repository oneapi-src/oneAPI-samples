# `EFI Application` Sample

##Building the EFI Application



| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04, Windows 10, Fedora 32, MacOS 10.15
| Hardware                          | Any machine which has an EFI based bios
| Software                          | Intel&reg; oneAPI C++ Compiler, Visual Studio, GCC-9, GCC-10, Clang-10

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

### Toolchains

For this, we assume you have a toolchain correctly installed.

We have tested on Windows with the Visual Studio 2019 compiler and the Intel&reg;C compiler, included with OneAPI.

On Linux, icc, clang and gcc are all supported.

#### Mac OS

To build on Mac OS - you will require a cross compiler to build the elf binaries (as mach-o binaries are not easily convertible to efi pe32+ format), and the debugger does not support MachO symbols. ma

The easiest way of getting this is to use homebrew (https://brew.sh/) and install the x86_64-elf-gcc formula.

```brew install x86_64-elf-gcc```


###build Process
To build the EFI application, please clone the repository.

```git clone https://github.com/oneapi-src/oneAPI-samples.git```

Then, go to the folder `Tools/SystemDebug/efi_appication`.


Then - clone all of the submodules

``git clone https://git.code.sf.net/p/gnu-efi/code  gnu-efi-code``
``git clone https://github.com/tianocore/edk2.git``
``git clone https://github.com/tianocore/edk2-BaseTools-win32.git Win32``

Now - make the cmake project, making sure to specify it is a 64-bit architecture.

```mkdir build```

```cd build```

To build the EFI application, we use the `cmake` build system to generate the build system.

We do this with:

```cmake ..```
> :warning: **If you are using Visual Studio 2017**: You will need to specify architecture. This can be done with `cmake .. -AX64`

Then - on windows, you will see a Visual Studio solution in the `build` directory, uxdbgapp.sln. On POSIX systems, you will see a MakeFile in the build folder.

You can then build this by issuing the following command:


```cmake --build .```

On Windows, you can open the visual studio solution and build the uxdbgapp target.

On POSIX, you can run the makefile with `make -j`.


#### __windows builders__

On windows, the output will be in the `Debug` folder in the build directory.


#### __Edk2 basetools__

On windows, you will either need to build edk2 basetools or get the prebuilt binaries.

The easiest way to get the prebuilt binaries is to clone the git repo.

``` cd <samples>/system_debugger/efi_application/efi/BaseTools/Bin```
``` git clone https://github.com/tianocore/edk2-BaseTools-win32.git Win32 ```

Alternatively, you can build the basetools yourself. For this, you will need to have visual studio installed and python.

First, you will have to add nmake to your PATH environment variable. NMAKE is located in the visual studio directory.
Then, install the edk2-pytool-library with

```pip install edk2-pytool-library```

Then, change to the edk2 basetools directory.

```cd <samples>/system_debugger/efi_application/edk2/BaseTools```

Then update all of the edk2 submodules.

```git submodule update --init --recursive```

Now, run the basetools build command. This is as follows:

```python Edk2ToolsBuild.py -t vs2019```

Where vs2019 is the Visual Studio 2019 compiler - For Visual Studio 2017, the command would be:


```python Edk2ToolsBuild.py -t vs2017```

If there are any errors, the build logs can be seen in:

```<samples>\system_debugger\efi_application\edk2\BaseTools\BaseToolsBuild\```

#### output files
This will output the following files:


##### Universal
`uxdbgapp.efi` - this is the efi application we need to flash to the device


##### Linux builders
`uxdbgapp.so` - this is the file containing the DWARF debug symbols on Linux

##### mac builders
`uxdbgapp.dylib` - this is the *ELF* file containing the DWARF debug symbols on mac OS.


##### Windows Builders
`uxdbgapp.pdb` - this is the file containing the CodeView debug symbols on windows

`uxdbgapp-te.efi` - this is the efi image in TERSE executable format

`uxdbgapp-te.pdb` - this is the file containing the CodeView debug symbols on windows for the Terse executable


## Flashing the EFI Application to a USB

### Linux and Mac

Copy the `uxdbgapp.efi` file into the `tools` folder.

Open a terminal, `cd` to the tools folder, and execute the script as follows:

`./make_boot_media.sh uxdbgapp.efi <dev/disk>`

##### NOTE

If you have issues with permissions, you might have to make the script executable.

you can do this by issuing the following command in the terminal:

```chmod +x make_boot_media.sh```

you can find information on the disks by using the following:

#### mac OS
```diskutil list```

#### linux
```sudo fdisk -l```


#### Windows

Start the PowerShell script from a PowerShell prompt.

If required, accept the access request for admin rights.

Then, select the .efi file using the file browser.

Next, select the USB device you would like to flash.

> :warning: **If you receive Execution Errors** You might have to change your Powershell execution policy. This can be done with the following command in PowerShell: `Set-ExecutionPolicy Unrestricted`

### Example Steps (Linux)

```
mkdir build
cd build
cmake ..
cmake --build .
chmod +x make_boot_media.sh
../make_boot_media.sh uxdbgapp.efi /dev/sdb
```



## Disclaimer
IMPORTANT NOTICE: This software is sample software. It is not designed or intended for use in any medical, life-saving or life-sustaining systems, transportation systems, nuclear systems, or for any other mission-critical application in which the failure of the system could lead to critical injury or death. The software may not be fully tested and may contain bugs or errors; it may not be intended or suitable for commercial release. No regulatory approvals for the software have been obtained, and therefore software may not be certified for use in certain countries or environments.


