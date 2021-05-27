# Intel oneAPI Toolkit Samples

The oneAPI-samples repository provides code samples for Intel oneAPI toolkits.

We recommend checking out a specific release version of the repository.
[View available releases](https://github.com/oneapi-src/oneAPI-samples/tags).

The latest versions of code samples on the master branch are not guaranteed to
be stable.

## Code Samples

|Code Sample    |Supported Intel(r)   Architecture(s)    	|Description 	|
|-----------------------|-------------------------------------------|---------------|
|__DirectPrograming/C++/__ |
|../C++/CombinationalLogic/Mandelbrot   	|CPU  	|Demonstrates how to accelerate   Mandelbrot performance with SIMD and parallelization using OpenMP*.   	|
|../C++/CompilerInfrastructure/Intrinsics    	|CPU  	|Shows how to utilize the   intrinsics supported by C++ compiler in a variety of applications.	|
|../C++/GraphTraversal/Mergesort   	|CPU  	|Shows how to accelerate scalar   merge sort program using OpenMP tasks   	|
|../Jupyter/OpenMP-offload-training | CPU |How to offload the computation to GPU using OpenMP with the Intel® C++ Compiler 
|../ParallelPatterns/openmp_reduction | GPU, CPU |How to run openMP on cpu as well as GPU offload
|../StructuredGrids/iso3dfd_omp_offload | CPU | How to offload the computation to GPU using Intel® oneAPI DPC++/C++ Compiler
|__DirectPrograming/DPC++/__ |
|../CombinationalLogic/mandelbrot 	|GPU, CPU  	|Example of a fractal in mathematics 	|
|../CombinationalLogic/sepia-filter    	|GPU, CPU  	|Color image conversion using 1D range    	|
|../DenseLinearAlgebra/complex_mult    	|GPU, CPU  	|Complex number Multiplication    	|
|../DenseLinearAlgebra/matrix_mul 	|GPU, CPU  	|Simple program that multiplies two large matrices in parallel using DPC++, OpenMP and MKL  	|
|../DenseLinearAlgebra/simple-add 	|FPGA, GPU, CPU 	|Simple Add program	|
|../DenseLinearAlgebra/vector-add 	|FPGA, GPU, CPU 	|Simple Vector add program   	|
|../GraphAlgorithms/all-pairs-shortest-paths | GPU, CPU | The All Pairs Shortest Paths sample demonstrates the following using the Intel® oneAPI DPC++/C++ Compiler|
|../GraphTraversal/bitonic-sort   	|GPU, CPU  	|Implementation of bitonic sort   using DPC++.   	|
|../Jupyter/oneapi-essentials-training   |GPU, CPU| Collection of Notebooks used ine the oneAPI Essentials training course|
|../MapReduce/MonteCarloPi   |GPU, CPU|How to utilize the DPC++ reduction extension|
|../N-BodyMethods/Nbody  |GPU, CPU| An N-body simulation is a simulation of a dynamical system of particles, usually under the influence of physical forces, such as gravity. This nbody sample code is implemented using C++ and DPC++ language for Intel CPU and GPU.|
|../ParallelPatterns/PreFixSum   	|GPU, CPU  	|Implement bitonic sort using Intel DPC++ compiler 	|
|../ParallelPatterns/dpc_reduce   	|GPU, CPU  	|A simple program that calculates   pi,  implemented using C++ and DPC++. 	|
|../ParallelPatterns/histogram   	|GPU, CPU  	|This sample demonstrates a histogram that groups numbers together and provides the count of a particular number in the input|
|../ParallelPatterns/loop-unroll   	|GPU, CPU  	|The Loop Unroll demonstrates a simple example of unrolling loops to improve the throughput of a DPC++ program for GPU offload.|
|../ProjectTemplatesmakefile-gpu | GPU |	A "Hello, world" Linux Makefile project for GPU |
|../ProjectTemplatesmakefile-fpga | FPGA |	A "Hello, world" Linux Makefile project for FPGA |
|../ProjectTemplatescmake-gpu | GPU |	A "Hello, world" Linux CMake project for GPU |
|../ProjectTemplatescmake-fpga | FPGA |	A "Hello, world" Linux CMake project for FPGA |
|../ProjectTemplatesHello_World_GPU | GPU |	A "Hello, world" Windows Visual Studio project for GPU |
|../SparseLinearAlgebra/merge-spmv { GPU, CPU | Sparse Matrix Vector sample provides a parallel implementation of a merge based sparse matrix and vector multiplication algorithm using DPC++. |
|../SpectralMethods/Discrete-cosine-transform    	|GPU, CPU  	|Image processing algorithm used   in JPEG compression	|
|../StructuredGrids/1d_HeatTransfer    	|GPU, CPU  	|A simulation of one dimensional   heat transfer process using DPC++.	|
|../StructuredGrids/ISO2DFD_DPCPP 	|GPU, CPU  	|A simple finite difference   stencil kernel for solving 2D acoustic isotropic wave equation using DPC++	|
|../StructuredGrids/ISO3DFD_DPCPP 	|GPU, CPU  	|A finite difference stencil   kernel for solving 3D acoustic isotropic wave equation using DPC++  	|
|../StructuredGrids/Particle-diffusion 	|GPU, CPU  	|A simple implementation of a   Monte Carlo simulation of the diffusion of water molecules in tissue
|__DirectPrograming/DPC++FPGA/ReferenceDesigns/__ |
|../crr                  |FPGA |High-performance CRR binomial tree option pricing model using DPC++ on FPGA |
|../db                   |FPGA |High-performance database query acceleration for FPGA-attached memory using DPC++ |
|../gzip                 |FPGA |High-performance GZIP compression using DPC++ on FPGA |
|../mvdr_beamforming     |FPGA |High-performance radar beamforming for streaming data using DPC++ on FPGA |
|../qrd                  |FPGA |High-performance QR decomposition of matrices using DPC++ on FPGA |
|__DirectPrograming/DPC++FPGA/Tutorials/Design Patterns__ |
|../buffered_host_streaming       |FPGA |Tutorial demonstrating how to create a high-performance full system CPU-FPGA design using SYCL USM. |
|../DesignPatterns/compute_units  |FPGA |Tutorial showing how to efficiently make multiple copies of a kernel, called compute units |
|../compute_units                 |FPGA |Ttutorial showcases a design pattern that allows you to make multiple copies of a kernel, called compute units. }               
|../double_buffering              |FPGA |Tutorial demonstrating how to overlap kernel execution with buffer transfers and host processing |
|../explicit_data_movement        |FPGA |Tutorial showing an alternative coding (explicit USM) in which data movement between host and device is controlled explicitly in code |
|../io_streaming                  |FPGA |Tutorial describing how to use DPC++ IO pipes to stream data through the FPGA's IO |
|../loop_carried_dependency       |FPGA |Tutorial demonstrating how to remove a loop-carried dependency to improve the performance of the FPGA device code |
|../n_way_buffering               |FPGA |Tutorial demonstrating how to extend double buffering to n-way buffering |
|../onchip_memory_cache           |FPGA |Tutorial explaining the caching of on-chip memory to reduce loop initiation interval on FPGA |
|../optimize_inner_loop           |FPGA |Tutorial explaining how to optimize the throughput of an inner loop with a low trip count |
|../pipe_array                    |FPGA |Tutorial showing how to create an array of pipes  |
|../shannonization                |FPGA |Tutorial explaining an optimization for removing computation from the critical path |
|../simple_host_streaming         |FPGA |Tutorial showing how to use SYCL Universal Shared Memory (USM) to stream data between the host and FPGA device and achieve low latency while maintaining throughput. |
|../triangular_loop               |FPGA |Tutorial explaining an advanced FPGA optimization technique for triangular loops |
|../zero_copy_data_transfer       |FPGA |Tutorial demonstrating how to use zero-copy host-device memory transfer for FPGA kernels with no temporal data reuse 
|__DirectPrograming//DPC++FPGA/Tutorials/Features__|
|../fpga_reg                            |FPGA |Tutorial demonstrating the use of the DPC++ FPGA power user extension intel::fpga_reg |
|../kernel_args_restrict                |FPGA |Tutorial demonstrating how to avoid performance penalties due to kernel argument aliasing |
|../loop_coalesce                       |FPGA |Tutorial demonstrating the DPC++ FPGA loop_coalesce attribute |
|../loop_initiation_interval            |FPGA |Tutorial demonstrating the use of the intel::initiation_interval attribute to improve performance |
|../loop_ivdep                          |FPGA |Tutorial demonstrating the use of the loop ivdep attribute |
|../loop_unroll                         |FPGA |Tutorial demonstrating the DPC++ unroll pragma and its performance trade-offs on FPGA |
|../lsu_control                         |FPGA |Tutorial demonstrating how to configure the load-store units (LSUs) in your DPC++ program for FPGA |
|../max_concurrency                     |FPGA |Tutorial demonstrating the DPC++ FPGA max_concurrency attribute |
|../max_interleaving                    |FPGA |Tutorial demonstrating the DPC++ FPGA max_interleaving attribute |
|../memory_attributes                   |FPGA |Tutorial demonstrating how to use DPC++ FPGA memory attributes |
|../pipes                               |FPGA |Tutorial demonstrating the DPC++ FPGA pipes extension to transfer data between kernels |
|../speculated_iterations               |FPGA |Tutorial demonstrating the DPC++ FPGA speculated_iterations attribute |
|__DirectPrograming/DPC++FPGA/Tutorials/GettingStarted__ |
|../fast_recompile                |FPGA |Tutorial introducing host-only recompile to save DPC++ development time on FPGA |
|../fpga_compile                  |FPGA |Tutorial introducing how to compile DPC++ for FPGA |
|__DirectPrograming/DPC++FPGA/Tutorials/tools__ |
|../dynamic_profiler                       |FPGA |Tutorial showing how to use the Intel® FPGA Dynamic Profiler for DPC++ for detailed FPGA kernel profiling|
|../system_profiling                       |FPGA |Tutorial showing how to use the OpenCL Intercept Layer to profile DPC++ designs running on FPGA at the system level |
|../use_library                            |FPGA |Tutorial showing how to use cross-language libraries in DPC++ on FPGA |
|__DirectProgramming/Fortran/__ |
|../CombinationalLogic/openmp-primes/ | GPU, CPU ||
|../DenseLinearAlgebra/optimize-integral/ | GPU, CPU | Optimization using the Intel® Fortran compiler |
|../DenseLinearAlgebra/vectorize-vecmatmult/ | GPU, CPU | Vectorization using Intel Fortran compiler |
|../Jupyter/OpenMP-offload-training/ | GPU, CPU | Collection of Jupyter notebooks that were developed to teach OpenMP Offload.|
|__Libraries__ |
|../oneCCL/tutorials/oneCCL_Getting_Started |GPU, CPU |port an Intel® oneAPI Collective Communications Library (oneCCL) sample from CPU to GPU|
|../oneDAL/IntelPython_daal4py_Distributed_Kmeans/|CPU| shows how to train and predict with a distributed k-means model using the python API package daal4py for oneAPI Data Analytics Library.|
|../oneDAL/IntelPython_daal4py_Distributed_LinearRegression/daal4py is a simplified API to Intel® DAAL that allows for fast usage of the framework suited for Data Scientists or Machine Learning users|
|../oneDAL/IntelPython_daal4py_Getting_Started | CPU |how how to do batch linear regression using the python API package daal4py from oneDAL.|
|../oneDPL/Gamma-correction 	|GPU, CPU  	|gamma correction using Parallel   STL 	|
|../oneDPL/Stable_sort_by_key    	|GPU, CPU  	|stable sort by key using   counting_iterator and zip_iterator  	|
|../oneVPL/hello-decode	|CPU  	|shows how to use oneVPL to   perform a simple video decode	|
|../oneVPL/hello-encode	|CPU  	|shows how to use oneVPL to   perform a simple video encode	|
|__Tools__ |
|../ApplicationDebugger/Debugger/array-transform    	|GPU, CPU  	|Array transform   	|
|../IoTConnectionTools/Analog-in	|CPU  	|Analog   pin input example using Eclipse* MRAA  	|
|../IoTConnectionTools/Digital   In  	|CPU  	|GPIO   pin input example using Eclipse* MRAA    	|
|../IoTConnectionTools/Digital   Out 	|CPU  	|GPIO   pin output example using Eclipse* MRAA   	|
|../IoTConnectionTools/Hello   IoT World  	|CPU  	|Basic   example that prints the compiler used during build	|
|../IoTConnectionTools/Interrupt	|CPU  	|Interrupt   Service Routine example using Eclipse* MRAA   	|
|../IoTConnectionTools/Onboard   Blink    	|CPU  	|Built-in   LED blink for common IoT boards using Eclipse* MRAA 	|
|../IoTConnectionTools/PWM 	|CPU  	|Pulse   Width Modulation pin output using Eclipse* MRAA   	|
|../IoTConnectionTools/Up2   LEDs    	|CPU  	|Built-in   LED example for UP* Squared using Eclipse* MRAA	|

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Known Issues or Limitations

### On Windows Platform
- If you are using Visual Studio 2019, Visual Studio 2019 version 16.4.0 or newer is required.
- Windows support for the FPGA code samples is limited to the FPGA emulator and optimization reports. Compile targets for FPGA hardware are provided on Linux only. See any FPGA code sample for more details.
- If you encounter a compilation error when building a sample program, such as the example error below, the directory path of the sample may be too long. The work around is to move the sample to a directory such as "c:\temp\sample_name".
    - Example error: *Error MSB6003 The specified task executable "dpcpp.exe" could not be run .......*

## Contribute

See [CONTRIBUTING](https://github.com/oneapi-src/oneAPI-samples/blob/master/CONTRIBUTING.md)
for more information.
