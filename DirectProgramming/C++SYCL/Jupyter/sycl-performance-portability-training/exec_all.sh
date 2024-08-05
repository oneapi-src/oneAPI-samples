#!/bin/bash
source /opt/intel/oneapi/setvars.sh

#Execute script

#matrix_size=("1024" "5120" "10240")
matrix_size=("2048")
device="DG1"
bin_dir="lab"
#bin_list=("mm_dpcpp_basic" "mm_dpcpp_ndrange" "mm_dpcpp_ndrange_var" "mm_dpcpp_localmem" "mm_dpcpp_mkl")
bin_list=("mm_dpcpp_localmem")


for bin in ${bin_list[*]}; do
  for size in ${matrix_size[*]}; do
     bin_args=" -n ${size}"
     echo ==================== >> exec_all_${device}.txt
     echo ${bin} >> exec_all_${device}.txt
     ./${bin_dir}/${bin}${bin_args} >> exec_all_${device}.txt
  done
done
    
