# Intel oneAPI Toolkit Samples

The oneAPI-samples repository provides code samples for Intel oneAPI toolkits.

We recommend checking out a specific release version of the repository.
[View available releases](https://github.com/oneapi-src/oneAPI-samples/tags).

The latest versions of code samples on the master branch are not guaranteed to
be stable.

## Code Samples

|Code Sample    |Supported Intel&reg;   Architecture(s)    	|Description 	|
|-----------------------|-------------------------------------------|---------------|
|__../AI-and-Analytics/End-to-end-Workloads/__|
|../Census |GPU, CPU| This sample code demonstrates how to seamlessly run the end-to-end census workload using the AI Analytics toolkit without any external dependencies. |
|../LidarObjectDetection-PointPillars |GPU, CPU | How to combine Intel® Distribution of OpenVINO™ toolkit and Intel® oneAPI to offload the computation of a complex workload to one of Intel's supported accelerators|
|__../AI-and-Analytics/Features-and-Functionality/__|
|../IntelPyTorch_Extensions_AutoMixedPrecision |CPU |You will learn how to download, compile, and get started with Intel Extension for PyTorch from this sample code.|
|../IntelPyTorch_TorchCCL_Multinode_Training |CPU | How to perform distributed training with oneCCL in PyTorch |
|../IntelPython_XGBoost_Performance |CPU | How to analyze the performance benefit from using Intel optimizations upstreamed by Intel to the latest XGBoost compared to un-optimized XGBoost 0.81|
|../IntelPython_XGBoost_daal4pyPrediction | CPU | How to analyze the performance benefit of minimal code changes to port pre-trained XGBoost model to daal4py prediction for much faster prediction than XGBoost prediction |
|../IntelPython_daal4py_DistributedKMeans |CPU |Distributed oneDAL K-Means programming model for Intel CPU
|../IntelPython_daal4py_DistributedLinearRegression |CPU |Distributed oneDAL Linear Regression programming model for Intel CPU |
|../IntelTensorFlow_Horovod_Multinode_Training |CPU |Shows how to get started with scaling out a neural network's training in TensorFlow on multiple compute nodes in a cluster.|
|../IntelTensorFlow_InferenceOptimization | CPU | Optimize a pre-trained model for a better inference performance
|../IntelTensorFlow_ModelZoo_Inference_with_FP32_Int8 |CPU | Show how to efficiently execute, train, and deploy Intel-optimized models |
|../IntelTensorFlow_PerformanceAnalysis | CPU |  Contains two Jupyter notebooks from Intel Model Zoo to help users analyze the performance difference between Stock Tensorflow and Intel Tensorflow.|
|__DirectPrograming/C++/__ |
|../CombinationalLogic/Mandelbrot   	|CPU  	|Demonstrates how to accelerate   Mandelbrot performance with SIMD and parallelization using OpenMP*.   	|
|../CompilerInfrastructure/Intrinsics    	|CPU  	|Shows how to utilize the   intrinsics supported by C++ compiler in a variety of applications.	|
|../GraphTraversal/Mergesort   	|CPU  	|Shows how to accelerate scalar   merge sort program using OpenMP tasks   	|
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
|../SparseLinearAlgebra/merge-spmv | GPU, CPU | Sparse Matrix Vector sample provides a parallel implementation of a merge based sparse matrix and vector multiplication algorithm using DPC++. |
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
|__DirectPrograming/DPC++FPGA/Tutorials/Design Patterns/__ |
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
|__DirectPrograming//DPC++FPGA/Tutorials/Features/__|
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
|__DirectPrograming/DPC++FPGA/Tutorials/GettingStarted/__ |
|../fast_recompile                |FPGA |Tutorial introducing host-only recompile to save DPC++ development time on FPGA |
|../fpga_compile                  |FPGA |Tutorial introducing how to compile DPC++ for FPGA |
|__DirectPrograming/DPC++FPGA/Tutorials/tools/__ |
|../dynamic_profiler                       |FPGA |Tutorial showing how to use the Intel® FPGA Dynamic Profiler for DPC++ for detailed FPGA kernel profiling|
|../system_profiling                       |FPGA |Tutorial showing how to use the OpenCL Intercept Layer to profile DPC++ designs running on FPGA at the system level |
|../use_library                            |FPGA |Tutorial showing how to use cross-language libraries in DPC++ on FPGA |
|__DirectProgramming/Fortran/__ |
|../CombinationalLogic/openmp-primes/ | GPU, CPU ||
|../DenseLinearAlgebra/optimize-integral/ | GPU, CPU | Optimization using the Intel® Fortran compiler |
|../DenseLinearAlgebra/vectorize-vecmatmult/ | GPU, CPU | Vectorization using Intel Fortran compiler |
|../Jupyter/OpenMP-offload-training/ | GPU, CPU | Collection of Jupyter notebooks that were developed to teach OpenMP Offload.|
|__Libraries/oneCCL/__ |
|../tutorials/oneCCL_Getting_Started |GPU, CPU |port an Intel® oneAPI Collective Communications Library (oneCCL) sample from CPU to GPU|
|__Libraries/oneDAL/__ |
|../IntelPython_daal4py_Distributed_Kmeans/|CPU| shows how to train and predict with a distributed k-means model using the python API package daal4py for oneAPI Data Analytics Library.|
|../IntelPython_daal4py_Distributed_LinearRegression  | CPU |daal4py is a simplified API to Intel® DAAL that allows for fast usage of the framework suited for Data Scientists or Machine Learning users|
|../IntelPython_daal4py_Getting_Started | CPU |how how to do batch linear regression using the python API package daal4py from oneDAL.|
|__Libraries/oneDNN/__ |
|../dpcpp_interoperability | GPU, CPU| emonstrates programming for Intel&reg; Processor Graphics with SYCL extensions API in oneDNN |
|../getting_started | GPU, CPU | Running a simple convolutional model on Intel CPU or Intel GPU |
|../simple_model | GPU, CPU | Running a simple convolutional model on Intel CPU or Intel GPU |
|../Tutorials |GPU, CPU | Series of Jupyer notebook tutorials on oneDNN |
|__Libraries/oneDPL/__ |
|../Gamma-correction 	|GPU, CPU  	|gamma correction using Parallel   STL 	|
|../Stable_sort_by_key    	|GPU, CPU  	|stable sort by key using   counting_iterator and zip_iterator  	|
|__Libraries/oneMKL/__ |
|../black_scholes |GPU, CPU|Shows how to use oneMKL's Vector Math (VM) and Random Number Generator (RNG) functionality to calculate the prices of options using the Black-Scholes formula for suitable randomly-generated portfolios.|
|../block_cholesky_decomposition|GPU, CPU|Shows how to use the oneMKL library's BLAS and LAPACK functionality to solve a symmetric, positive-definite block tridiagonal linear equation.|
|../block_lu_decomposition|GPU, CPU|Shows how to use the oneMKL library's BLAS and LAPACK functionality to solve a block tridiagonal linear equation.|
|../computed_tomography|GPU, CPU|Shows how to use the oneMKL library's DFT functionality to simulate computed tomography (CT) imaging.
|../fourier_correlation|GPU, CPU|Shows how to implement a 1D Fourier correlation using oneMKL kernel functions.
|../matrix_mul_mkl|GPU, CPU|Shows how to use the oneMKL's optimized matrix multiplication routines.|
|../monte_carlo_european_opt|GPU, CPU|Shows how to use the oneMKL library's random number generation (RNG) functionality to compute European option prices.|
|../monte_carlo_pi|GPU, CPU|Shows how to use the oneMKL library's random number generation (RNG) functionality to estimate the value of π.|
|../random_sampling_without_replacement|GPU, CPU|hows how to use the oneMKL library's random number generation (RNG) functionality to generate K>>1 simple random length-M samples without replacement |
|../sparse_conjugate_gradient|GPU, CPU|Shows how to use the oneMKL library's sparse linear algebra functionality to solve a sparse, symmetric linear system using the (preconditioned) conjugate gradient method.|
|../student_t_test|GPU, CPU|Shows how to use the oneMKL library's Vector Statistics functionality to decide if the null hypothesis should be accepted or rejected.|
|__Libraries/oneTBB/__ |
|../tbb-async-sycl |GPU, CPU|Show the computational kernel can be split for execution between CPU and GPU using TBB Flow Graph asynchronous node and functional node. |
|../tbb-resumable-tasks-sycl |GPU, CPU|Show the computational kernel can be split for execution between CPU and GPU using TBB resumable task and parallel_for. 
|../tbb-task-sycl |GPU, CPU|Show two TBB tasks can execute similar computational kernels, with one task executing the SYCL code and the other task executing the TBB code. 
|__Libraries//oneVPL/__ |
|../dpcpp-blur	|CPU  	|How to use oneVPL and DPC++ to convert I420 raw video files into BGRA and blur each frame|
|../hello-decode	|CPU  	|shows how to use oneVPL to   perform a simple video decode	|
|../hello-encode	|CPU  	|shows how to use oneVPL to   perform a simple video encode	|
|../oneVPL/hello-vpp	|CPU  	|How to use oneVPL to resize an I420 raw video file	|
|__Tools/ApplicationDebugger/__|
|../Debugger/array-transform    	|GPU, CPU  	|Array transform   	|
|__Tools/IoTConnectionTools/__| 
|../Analog-in	|CPU  	|Analog   pin input example using Eclipse* MRAA  	|
|../aws-pub-sub	|CPU  	|A sample that could be used for a quick test of Amazon cloud libraries.  	|
|../azure-iothub-telemetry|CPU|A sample demonstrating how to send messages from a single device to Microsoft Azure IoT Hub via a selected protocol.
|../digital-in  	|CPU  	|GPIO pin input example using Eclipse* MRAA    	|
|../digital-out 	|CPU  	|GPIO pin output example using Eclipse* MRAA   	|
|../hello-iot-world  	|CPU  	|Basic   example that prints the compiler used during build	|
|../ibm-device   |CPU  	|CPU	| A sample shows how to develop a device code using Watson IoT Platform iot-c device client library, connect and interact with Watson IoT Platform Service.|
|../Interrupt	|CPU  	|Interrupt   Service Routine example using Eclipse* MRAA   	|
|../Onboard-Blink    	|CPU  	|Built-in   LED blink for common IoT boards using Eclipse* MRAA 	|
|../PWM 	|CPU  	|Pulse   Width Modulation pin output using Eclipse* MRAA   	|
|../Up2   LEDs    	|CPU  	|Built-in   LED example for UP* Squared using Eclipse* MRAA	|
|__Tools/Migration/__|
|../folder-options-dpct |CPU| How to migrate more complex projects and use options for DPC++ Compatibility Tool (dpct) |
|../rodinia-nw-dpct |CPU|How to migrate a Make/CMake project from CUDA to Data Parallel C++ using the Intel DPC++ Compatibility Tool. |
|../vector-add-dpct |CPU|how to migrate a simple program from CUDA to Data Parallel C++ using the Intel DPC++ Compatibility Tool. |
|__Tools/VTuneProfiler__/
|../matrix_multiply_vtune|CPU|A sample containing multiple implementations of matrix multiplication. This sample code is implemented using DPC++ language for CPU and GPU.|

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
