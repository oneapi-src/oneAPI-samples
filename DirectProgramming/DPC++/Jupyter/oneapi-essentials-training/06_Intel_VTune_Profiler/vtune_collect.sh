#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module6 -- Intel Vtune profiler - 1 of 1 Vtune_Profiler
#vtune
#type=hotspots
#type=memory-consumption
#type=uarch-exploration
#type=memory-access
#type=threading
#type=hpc-performance
#type=system-overview
#type=graphics-rendering
#type=io
#type=fpga-interaction
#type=gpu-offload
type=gpu-hotspots
#type=throttling
#type=platform-profiler
#type=cpugpu-concurrency
#type=tsx-exploration
#type=tsx-hotspots
#type=sgx-hotspots

rm -r vtune_data

echo "VTune Collect $type"
vtune -collect $type -result-dir vtune_data $(pwd)/iso3dfd 256 256 256 8 8 8 20 sycl gpu

echo "VTune Summary Report"
vtune -report summary -result-dir vtune_data -format html -report-output $(pwd)/summary.html
