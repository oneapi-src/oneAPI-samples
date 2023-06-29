Matrix multiplication kernel
----------------------------
You can select one among several multiply kernels available in the sample. Specify one of them as a macro definition of MULTIPLY in the multiply.h header file and rebuild the project.

multiply0 - Basic serial implementation
multiply1 - Basic multithreaded implementation
multiply2 - Optimized implementation with Loop interchange and vectorization (use Compiler vectorization options)
multiply3 - Optimized implementation with adding Cache blocking and data alignment (add ALIGNED macro to Compiler preprocessor)
multiply4 - Optimized implementation with matrix transposition and loop unrolling
multiply5 - Most optimal version with using Intel MKL (link the MKL library to the project and add the USE_MKL macro to Compiler preprocessor)

Note: multiply kernels availability depends on threading model you choose.

Threading
---------
Default threading model is native, i.e. pthreads on Linux and Win32 threads on Windows.
OpenMP threading model is available.
For the MKL-based kernel consider either MKL's multithreaded or singlethreaded implementation. Make the kernel multithreaded by yourself in the latter case. 

In order to change the threading model
In Windows project: select either of Release (default), Release_OMP, or Release_MKL configuration.
In Linux Makefile: select PARAMODEL either of USE_THR, USE_OMP, or USE_MKL.

Matrix properties
-----------------
By default a square matrix is used with size of 2048. You may want to increase matrix size by redefining the macro MAXTHREADS in the multiply.h. However, make sure the size is multiple of # of threads. It's made for simplicity.
By default # of threads executed is equal to number of CPU cores available in the system (defined in run-time). You may want to limit max number of created threads by modifying the MAXTHREADS.

Building the matrix sample
--------------------------
Linux:
Use the Makefile in the linux subdirectory and specify compiler you want to use: icc or gcc (see the comments inside the Makefile)

Windows:
Use the matrix.sln solution in the vc9 subdirectory. Convert the solution and project for later version of MSFT VS if necessary. 
Intel compiler is a default configuration in the solution. 

Building for MIC co-processor
-----------------------------
Linux:
Use the Makefile in the linux subdirectory and specify mic option (make mic)

Windows:
Use the buildmatrix.bat batch file from the windows_mic subdirectory (see the comments inside the buildmatrix.bat). 
