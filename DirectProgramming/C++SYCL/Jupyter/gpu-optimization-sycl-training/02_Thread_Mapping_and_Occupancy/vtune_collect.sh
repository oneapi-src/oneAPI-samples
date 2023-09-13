#!/bin/bash
source /opt/intel/oneapi/setvars.sh

# Compile code
icpx -fsycl lab/vec_add.cpp -o vec_add

#Vtune GPU Hotspot script

bin="vec_add"
prj_dir="vtune_data"

echo $bin
rm -r ${prj_dir}
echo "Vtune Collect GPU hotspots"
vtune -collect gpu-hotspots -result-dir ${prj_dir} $(pwd)/${bin}



