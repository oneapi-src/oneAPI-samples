#!/bin/bash
source /opt/intel/oneapi/setvars.sh

#Vtune GPU Hotspot script

#matrix_size=("1024" "5120" "10240")
matrix_size=("5120")
wg_size="16"
device="DG1"
bin_dir="lab"

#bin_list=("mm_dpcpp_basic" "mm_dpcpp_ndrange" "mm_dpcpp_ndrange_var" "mm_dpcpp_localmem" "mm_dpcpp_mkl")
bin_list=("mm_dpcpp_localmem")
prj_dir="vtune_data"

for bin in ${bin_list[*]}; do
  for size in ${matrix_size[*]}; do
     bin_args=" -n ${size} -m ${wg_size}"
     echo ====================
     echo $bin
     rm -r ${prj_dir}
     echo "Vtune Collect hotspots"
     # For GPU Collection only uncomment below
     vtune -collect gpu-hotspots -result-dir ${prj_dir} $(pwd)/${bin_dir}/${bin}${bin_args}
     # For CPU Collection only uncomment below
     #vtune -collect hotspots -knob sampling-mode=hw -result-dir ${prj_dir} $(pwd)/${bin_dir}/${bin}${bin_args}
     echo "Vtune Summary Report"
     vtune -report summary -result-dir ${prj_dir} -format html -report-output $(pwd)/vtune_${bin}_${device}_${size}.html
  done
done

