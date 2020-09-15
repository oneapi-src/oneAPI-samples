###########################
``EFI Application`` Sample
###########################

============================
Building the EFI Application
============================

+-----------------+-----------------------------------------------------------------------------+
| Optimized for   | Description                                                                 |
+=================+=============================================================================+
| OS              | Linux\* Ubuntu\* 18.04, Windows 10, Fedora 32, MacOS 10.15                  |
+-----------------+-----------------------------------------------------------------------------+
| Hardware        | Any machine which has an EFI based bios (UpXtreme detailed here)            |
+-----------------+-----------------------------------------------------------------------------+
| Software        | Intel® oneAPI C++ Compiler (beta), Visual Studio, GCC-9, GCC-10, Clang-10   |
+-----------------+-----------------------------------------------------------------------------+

License
-------

This code sample is licensed under MIT license.

Hardware
------------

For this sample, we recommend you use the `Aaeon Upxtreme <https://www.aaeon.com/en/p/up-xtreme-compact-embedded-board-whiskey-lake>`_.

This is a Whiskeylake based design.


To perform source level debugging, you will have to flash the debug bios on the device.

This can be done by downloading the debug firmware package `Here <https://downloads.up-community.org/download/up-xtreme-uefi-debug-bios-v1-8d/>`_.

Then, put these files to a USB stick, boot the UpXtreme to an EFI shell (more detailed instructions detailed on above link), and run the GO_entire.nsh script.

Now - reboot the target and press F2 to go into the bios settings. You will see a password page.

Please enter the default password **upassw0rd**

First, lets change the boot order to boot from USB first. Go to the **BOOT** section in the bios, and change *Boot Option #1* to your desired USB stick.


Then, to enable debugging, go to **CRB Setup** -> **CRB Advanced** -> **Debug Settings**

Then, enable the following:

**Platform Debug Consent** - change this to *Enabled (usb3 Dbc)*

Then, go to Advanced Debug Settings:

Make sure **CPU Run Control** is enabled. And enable processor trace memory allocation to **128MB**. Then set **Processor Trace** to enabled.
Then, finally make sure **Processor Trace output Scheme** is set to **Single Range Output**.

Now, go back (by pressing esc) to the CRB main menu. Go to **CRB Save & Exit** and click **Save Changes and Exit**.

Now, you can plug the DbC cable into your Host and the UpXtreme. The DBC cable is a Usb 3 cable, with no power pins. You can buy a cable
`from Design in Tools <https://designintools.intel.com/Design-Validation-Tools-Design-In-Tools-a/287.htm/>`_.





Toolchains
--------------

for this, we assume you have a toolchain correctly installed.

On windows, we have tested with the Visual Studio 2019 compiler, and the
Intel®C compiler, included with OneAPI.

On linux, icc , clang and gcc are all supported.

Mac OS
^^^^^^

To build on mac OS - you will require a cross compiler to build the elf
binaries (as mach-o binaries are not easily convertable to efi pe32+
format), and the debugger does not support MachO symbols. ma

The easiest way of getting this, is to use homebrew (https://brew.sh/)
and install the x86\_64-elf-gcc formula.

``brew install x86_64-elf-gcc``

build Process
--------------

To build the efi applcation, please clone the repository.

``git clone https://github.com/oneapi-src/oneAPI-samples.git``

Then, cd to the folder `Tools/SystemDebug/`.

Then cd to the location of the efi application.

``cd system_debug_sample_build/efi_application/``

Then - clone all of the submodules

``git submodule update --init``

Edk2 basetools
^^^^^^^^^^^^^^^^

On windows, you will either need to build edk2 basetools, or get the
prebuilt binaries.

The easiest way to get the prebuilt binaries is to clone the git repo.

| ``cd <samples>/system_debugger/efi_application/efi/BaseTools/Bin``

| ``git clone https://github.com/tianocore/edk2-BaseTools-win32.git Win32``

Alternatively, you can build the basetools yourself. For this, you will
need to have visual studio installed, and python.

First, you will have to add nmake to your PATH environment variable.
NMAKE is located in the visual studio directory.


| Then, install the edk2-pytool-library with

``pip install edk2-pytool-library``

Then, change to the edk2 basetools directory.

``cd <samples>/system_debugger/efi_application/edk2/BaseTools``

Then update all of the edk2 submodules.

``git submodule update --init --recursive``

Now, run the basetools build command. This is as follows:

``python Edk2ToolsBuild.py -t vs2019``

Where vs2019 is the visual studio 2019 compiler - For vs2017 the command
would be:

``python Edk2ToolsBuild.py -t vs2017``

If there are any errors, the build logs can be seen in:

``<samples>\system_debugger\efi_application\edk2\BaseTools\BaseToolsBuild\``


Building
^^^^^^^^^^^^^^^^

Now - make the cmake project, making sure to specify it is a 64 bit
architecture.

``mkdir build``

``cd build``

To build the efi application, we use the ``cmake`` build system to
generate the build system.

We do this with:

``cmake ..``

.. note::
    On Windows, when using Visual Studio 2017 or older, you will need to specify the architecture.
    Please use the following cmake command:

    ``cmake .. -AX64``

then - on windows you will see a Visual Studio solution in the ``build``
directory, uxdbgapp.sln, on POSIX systems you will see a MakeFile in the
build folder.

You can then build this by issuing the following command:

``cmake --build .``

Or,

on Windows, you can open the visual studio solution, and build the
uxdbgapp target.

On POSIX, you can run the makefile with ``make -j``.

windows builders
^^^^^^^^^^^^^^^^
On windows, the output will be in the ``Debug`` folder in the build
directory.


output files
------------

This will output the following files:

Universal
^^^^^^^^^

``uxdbgapp.efi`` - this is the efi appication we need to flash to the
device

linux builders
^^^^^^^^^^^^^^

``uxdbgapp.so`` - this is the file containing the DWARF debug symbols on
linux

mac builders
^^^^^^^^^^^^
``uxdbgapp.dylib`` - this is the *ELF* file containing the DWARF debug
symbols on mac OS.


Windows Builders
^^^^^^^^^^^^^^^^

``uxdbgapp.pdb`` - this is the file containing the CodeView debug
symbols on windows

``uxdbgapp-te.efi`` - this is the efi image in TERSE executable format

``uxdbgapp-te.pdb`` - this is the file containing the CodeView debug
symbols on windows for the Terse executable


Flashing the EFI Application to a USB
------------------------------------------

Linux and Mac
^^^^^^^^^^^^^^

Copy the ``uxdbgapp.efi`` file into the ``tools`` folder.

Open a terminal, ``cd`` to the tools folder, and execute the script as
follows:

``./make_boot_media.sh uxdbgapp.efi <dev/disk>``


.. note::
    if you have issues with permissions, you might have to make the script
    executable.

    you can do this by issueing the following command in terminal:

    ``chmod +x make_boot_media.sh``

you can find information on the disks by using the following:

mac OS
^^^^^^^^^^^^^^


``diskutil list``

linux
^^^^^^^^^^^^^^


``sudo fdisk -l``

Windows
^^^^^^^^^^^^^^

Start the powershell script, from a powershell prompt.

If required, accept the access request for admin rights.

Then, select the .efi file using the file browser.

Next, select the usb device you would like to flash to.

Example Steps (Linux)
------------------------

::

    mkdir build
    cd build
    cmake ..
    cmake --build .
    chmod +x make_boot_media.sh
    ../make_boot_media.sh uxdbgapp.efi /dev/sdb

if you recieve execution errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You might have to change your powershell execution policy. This can be
done with the following comman in powershell:
``Set-ExecutionPolicy Unrestricted``

Disclaimer
----------

IMPORTANT NOTICE: This software is sample software. It is not designed
or intended for use in any medical, life-saving or life-sustaining
systems, transportation systems, nuclear systems, or for any other
mission-critical application in which the failure of the system could
lead to critical injury or death. The software may not be fully tested
and may contain bugs or errors; it may not be intended or suitable for
commercial release. No regulatory approvals for the software have been
obtained, and therefore software may not be certified for use in certain
countries or environments.
