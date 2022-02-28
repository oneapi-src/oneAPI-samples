# Intel® DPC++ Compatibility Tool: Foo Bar Example

## Use the Command Line to Migrate Large Code Bases

The Intel DPC++ Compatibility Tool (dpct) can migrate projects that include
multiple source and header files. This sample provides an example of how to
migrate more complex projects and use options.


| Optimized for         | Description
|:---                   |:---
| OS                    | Linux* Ubuntu* 18.04; Windows 10
| Software              | Intel DPC++ Compatibility Tool;
| What you will learn   | Simple invocation of dpct to migrate CUDA code
| Time to complete      | 10 minutes


## Purpose

The Intel® DPC++ Compatibility Tool (dpct) can be used to migrate projects
composed of multiple source and header files. This includes headers from the
system libraries and other projects. The tool must know which headers need
migration and which should be left alone.

Use the dpct `--in-root` option to set the root location of your program
sources that are to be migrated. Only files and folders located within the
--in-root directory will be considered for migration by the tool. Files located
outside the`--in-root` directory are considered system files and will not be
migrated, even if they are included by a source file located within the
`--in-root`directory.

The dpct `--out-root` option specifies the directory into which the DPC++ code
produced by the Intel DPC++ Compatibility Tool is written. The relative
location and names of the migrated files are maintained, except the file
extensions are changed to `.dp.cpp`.


## Key Implementation Details

Use --in-root and --out-root for projects which contain more than one source
file.  Additional migration options can be reviewed at:
[Command Line Options Reference](https://software.intel.com/content/www/us/en/develop/documentation/intel-dpcpp-compatibility-tool-user-guide/top/command-line-options-reference.html).


## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Using Visual Studio Code* (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.
 - (Linux only) Debug your GPU application with GDB for Intel® oneAPI toolkits using the **Generate Launch Configurations** extension.

To learn more about the extensions, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

## Migrating the CUDA Sample to Data Parallel C++ with the Intel DPC++ Compatibility Tool

Building and running the CUDA sample is not required to migrate this project
to a Data Parallel C++ project.

> **NOTE:** Certain CUDA header files, referenced by the CUDA application
> source files to be migrated, need to be accessible for the migration step.
> See the [Getting Started Guide][cuda-headers] for more details.

[cuda-headers]: <https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-intel-dpcpp-compatibility-tool/top.html#top_BEFORE_YOU_BEGIN>

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux Sudo: . /opt/intel/oneapi/setvars.sh
>
> Linux User: . ~/intel/oneapi/setvars.sh
>
> Windows: C:\Program Files(x86)\Intel\oneAPI\setvars.bat
>
>For more information on environment variables, see Use the setvars Script for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### Command-Line on a Linux* System

1. This sample project contains a simple CUDA program with three files
   (main.cu, util.cu and util.h) located in two folders (foo and bar):

```
 foo
 ├── bar
 │   ├── util.cu
 │   └── util.h
 └── main.cu
```

2. Use the tool's `--in-root` option and provide input files to specify where
   to locate the CUDA files that need migration; use the tool’s `--out-root`
   option to designate where to generate the resulting files:

```sh
# From the repo root directory:
$ dpct --in-root=foo --out-root=result/foo foo/main.cu foo/bar/util.cu --extra-arg="-Ifoo/bar/"
```

> If an `--in-root` option is not specified, the directory of the first input
> source file is implied. If `--out-root` is not specified, `./dpct_output`
> is implied.

You should see the migrated files in the `result/foo` folder that was specified
by the `--out-root` option:

```
 result/foo
        ├── bar
        │   ├── util.dp.cpp
        │   └── util.h
        └── main.dp.cpp
```

3. Inspect the migrated source code, address any `DPCT` warnings generated
   by the Intel DPC++ Compatibility Tool, and verify the new program correctness.

Warnings are printed to the console and added as comments in the migrated
source. See the [Diagnostic Reference][diag-ref] for more information on what
each warning means.

[diag-ref]: <https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-intel-dpcpp-compatibility-tool/top/diagnostics-reference.html>

This sample should generate the following warning:

```
warning: DPCT1015:0: Output needs adjustment.
```

See below **Addressing Warnings in the Migrated Code** to understand how to
resolve the warning.


4. Copy the original `Makefile` into the `result` folder and update the
   copy to build the migrated project using DPC++. Replace the CUDA
   configurations in that new `Makefile` with the following for use with DPC++:

```Makefile
CXX = dpcpp

# Remainder of the Makefile should work without changes.
```

> **NOTE:** The above Makefile changes work for this sample project. The
> modifications needed to update the build files in your own projects will vary
> greatly depending on the nature and complexity of your migrated projects.

5. Switch to the migration directory with `cd result`.

6. Build the migrated sample with `make`.

7. Run the migrated sample with `make run`.

8. Clean up the build with `make clean`.


## Windows

1. This sample project contains a simple CUDA program with three files
   (main.cu, util.cu and util.h) located in two folders (foo and bar):

```
 foo
 ├── bar
 │   ├── util.cu
 │   └── util.h
 └── main.cu
```

2. Use the dpct's `--in-root` and `--out-root` options to specify where to
   locate the migrated CUDA files:

```bat
> dpct --in-root=foo --out-root=result\foo foo\main.cu foo\bar\util.cu --extra-arg="-Ifoo\bar\"
```

This sample should generate the following warning:

```
warning: DPCT1015:0: Output needs adjustment.
```

See below **Addressing Warnings in the Migrated Code** to understand how to
resolve the warning.


> If an `--in-root` option is not specified, the directory of the first input
> source file is implied. If `--out-root` is not specified, `./dpct_output`
> is implied.

You should see the migrated files in the `result/foo` folder that was specified
by the `--out-root` option:

```
result/foo
        ├── bar
        │   ├── util.dp.cpp
        │   └── util.h
        └── main.dp.cpp
```

To build this migrated application on Windows, you must modify the
original `Makefile` to be compatible with Microsoft `nmake` and Windows
command-line tools.


# Addressing Warnings in Migrated Code

Migration generated one warning for code that `dpct` could not migrate:

```
warning: DPCT1015:0: Output needs adjustment.
```

As you have noticed, the migration of this project resulted in one DPCT
message that needs to be addressed, DPCT1015. This message is shown because as
the Compatibility Tool migrated from the printf-style formatted string in the
CUDA code to the output stream supported by DPC++, manual adjustment is needed
to generate the equivalent output.

Open result/foo/bar/util.dp.cpp and locate the error  DPCT1015. Then make the
following changes:

Change:
```
stream_ct1 << "kernel_util,%d\n";
```
to
```
stream_ct1 << "kernel_util," << c << sycl::endl;
```

You’ll also need to change the stream statement in result/foo/main.dp.cpp.

Change:
```
stream_ct1 << z"kernel_main!\n";
```
to
```
stream_ct1 << "kernel_main!" << sycl::endl;
```

# Example Output

When you run the migrated application, you should see the following console
output:

```
./foo-bar
kernel_main!
kernel_util,2
```

> **NOTE:** If you see the following TBB error message, you can safely it.
> Not all users will see this TBB error message. It has nothing to do with the
> migration of your application and does not mean that your application is not
> running correctly. The issue will be resolved in a future oneAPI release.

```
TBB Warning: The number of workers is currently limited to 11. The request for 31 workers is ignored. Further requests for more workers will be silently ignored until the limit changes.
```

If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)