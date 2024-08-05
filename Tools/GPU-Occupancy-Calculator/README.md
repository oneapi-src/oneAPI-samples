# Intel(R) GPU Occupancy Calculator

## Purpose

Intel GPU Occupancy Calculator is HTML Web Application to compute GPU Occupancy
for GPU/HPC application development.

## Tool Details

The Intel GPU Occupancy tool will compute the theoretical GPU Occupancy for Intel GPU devices based on GPU Kernel code parameters. The Occupancy value will determine whether the GPU is efficiently utilizing all the GPU hardware resources. The tool will also generate graphs that will help optimize GPU Occupancy.

* Allows user to select a GPU, input Global Size of HPC application,
  Work-Group(WG) size, Sub-Group(SG) size and Shared Local Memory(SLM) size.
* Computes Theoretical Occupancy based on the inputs.
* Generates charts for Impact of varying Work-Group size, Impact of varying Shared Local Memory size and GPU Occupancy graph for the workload.

## Usage
* Open Page: [oneapi-src.github.io/oneAPI-samples/Tools/GPU-Occupancy-Calculator/](https://oneapi-src.github.io/oneAPI-samples/Tools/GPU-Occupancy-Calculator/)
* Select a GPU from the dropdown and change "XVE Count" if necessary OR select
  the option to enter PCI ID for GPU.
* The tool will load some default values for Global Size, Work-Group size, Sub-Group size,
  Local Memory size and will compute Occupancy and generate graphs.
* Change the values of Global size, Work-Group size, Sub-Group size or Local Memory size based on your HPC application to calculate Occupancy and tune application.

  
### Custom GPU Configuration:
If the GPU drop down does not have the configuration, a custom configuration can be loaded using URL parameter "`?gpu=`"

An example of a custom configuration and corresponding description are listed below:

Example of Intel GPU Occupancy Calculator with Custom GPU configuration:
  
`https://oneapi-src.github.io/oneAPI-samples/Tools/GPU-Occupancy-Calculator/?gpu=8;8;512;64;true;[32,16];128;128;[0,1,2,4,8,16,24,32,48,64,96,128];1024;64;64;My_GPU`
  
URL Parameters details are as follows:
```  
  EU_Per_XeCore = 8
  Threads_Per_EU = 8
  Total_EU_Count = 512
  Max_Threads_Per_XeCore = 64
  Large_GRF_Mode_available = true
  Subgroup_Sizes = [32, 16]
  SLM_Size_Per_XeCore = 128
  SLM_Size_Per_Work_Group = 128
  TG_SLM_Sizes = [0,1,2,4,8,16,24,32,48,64,96,128]
  Max_Work_Group_Size = 1024
  Max_Num_Of_Workgroups = 64
  Max_Num_Of_Barrier_Registers = 64
  Custom_GPU_Name (Optional) = My_GPU
```
## License

Code samples are licensed under the MIT license. See _License.txt_ for details.

Third party program Licenses can be found in _third-party-programs.txt_
