# Intel(R) GPU Occupancy Calculator

## Purpose

Intel GPU Occupancy Calculator is HTML Web Application to compute GPU Occupancy for GPU/HPC application development.

## Tool Details

* Allows user to pick a GPU SKU, input Global Size of HPC application, Work-Group(WG) size, Sub-Group size, Shared Local Memory(SLM) size and barrier usage.
* Computes Sub-Slice/Dual Sub-Slice (SS/DSS) Theoretical Occupancy based on the inputs.
* Generates charts for Impact of varying Work-Group size and Impact of varying Shared Local Memory size.
* Generates a list of all valid Work-Group sizes that can be used and corresponding Occupancy.
* Generates a list of optimal configuration values for WG, SG and SLM limit that will get 100% Occupancy.

## Usage
* Open Page: [oneapi-src.github.io/oneAPI-samples/Tools/GPU-Occupancy-Calculator/](https://oneapi-src.github.io/oneAPI-samples/Tools/GPU-Occupancy-Calculator/)
* Select a GPU from the dropdown and change "EU Count" if necessary OR select the option to enter PCI ID for GPU.
* The tool will load some default values for  Global Size, WG size, SG size, SLM size and will compute Occupancy and generate graphs.
* Change the values of GPU target, Global size, WG size, SG size, SLM size or Barrier usage based on your HPC application to calculate Occupancy and tune application.

## License

Code samples are licensed under the MIT license. See _License.txt_ for details.

Third party program Licenses can be found in _third-party-programs.txt_
