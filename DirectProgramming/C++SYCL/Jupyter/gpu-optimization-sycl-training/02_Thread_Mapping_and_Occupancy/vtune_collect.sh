#!/bin/bash
source /opt/intel/oneapi/setvars.sh

# Compile code
dpcpp lab/vec_add.cpp -o vec_add

#Vtune GPU Hotspot script

device="gen9"
bin="vec_add"
prj_dir="vtune_data"

echo $bin
rm -r ${prj_dir}
echo "Vtune Collect hotspots"
vtune -collect gpu-hotspots -result-dir ${prj_dir} $(pwd)/${bin}
 echo "Vtune Summary Report"
 vtune -report summary -result-dir ${prj_dir} -format html -report-output $(pwd)/vtune_${bin}_${device}.html


