# ISO3DFD sample
ISO3DFD is a finite difference stencil kernel for solving 3D acoustic isotropic wave equation which can be used as a proxy for propogating a seismic wave. Kernels in this sample are implemented as 16th order in space, with symmetric coefficients, and 2nd order in time scheme without boundary conditions. This sample code is implemented using Data Parallel C++ for CPU and GPU and using OpenMP on CPU for comparison and validation.
  
| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux Ubuntu 18.04; Windows 10 or Windows Server 2017
| Hardware                          | Kaby Lake with GEN9 or newer
| Software                          | Intel Data Parallel C++ Compiler beta; Intel C++ Compiler xxx beta
| What you will learn               | How to offload the computation to GPU using Intel DPC++ compiler and OpenMP offload pragmas
| Time to complete                  | 15 minutes

Performance number tabulation [Dummy data for now]

| iso3dfd                           | Performance data
|:---                               |:---
| Scalar baseline -O2               | 1.0

  
## Key implementation details [optional]
SYCL implementation explained. 

You're encourage to use any infographics to increase readibility and clarity which is strongly recommended by UX team based on customer feedbacks. 

## How other tools (Intel libraries or Intel tools) are used [optional]
   

## License  
This code sample is licensed under MIT license under xxx  

## How to Build  

### on Linux  
   * Build iso3dfd  
    
    cd iso3dfd &&  
    mkdir build &&  
    cd build &&  
    cmake .. &&  
    make 

   * Run the program on Gen9 
    
    make run  
   
   * Run the program on CPU 
    
    make run_cpu  

   * Clean the program  
    make clean  

### on Windows

INSTRUCTIONS HERE. EXAMPLE:

## How to Run  
   * Application Parameters   
	Usage: ./iso3dfd n1 n2 n3 b1 b2 b3 Iterations [omp|sycl] [gpu|cpu]
	
	n1 n2 n3      : Grid sizes for the stencil
	b1 b2 b3      : cache block sizes for cpu openmp version.
	Iterations    : No. of timesteps.
	[omp|sycl]    : Optional: Run the OpenMP or the SYCL variant. Default is to use both for validation
	[gpu|cpu]     : Optional: Device to run the SYCL version Default is to use the GPU if available, if not fallback to CPU


## Performance Tests
   * DPC++ on CPU
	
    numactl -c 0 -m 0 ./iso3dfd 256 256 256 256 8 8 100 sycl cpu   
   * DPC++ on Gen9   
	
    ./iso3dfd 256 256 256 256 8 8 100 sycl gpu   
	  

## Validation Tests
   * DPC++ on CPU
	
    numactl -c 0 -m 0 ./iso3dfd 256 256 256 256 8 8 10 cpu   
   * DPC++ on Gen9   
    
    ./iso3dfd 256 256 256 256 8 8 10 gpu   
