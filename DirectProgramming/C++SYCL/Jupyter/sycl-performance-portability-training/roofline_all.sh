#!/bin/bash
source /opt/intel/oneapi/setvars.sh

#Advisor Roofline script

#matrix_size=("1024" "5120" "10240")
matrix_size=("5120")
wg_size="16"
device="Gen9"
bin_dir="lab"
#bin_list=("mm_dpcpp_ndrange" "mm_dpcpp_ndrange_var" "mm_dpcpp_localmem" "mm_dpcpp_mkl")
bin_list=("mm_dpcpp_localmem")
prj_dir="./roofline_data"


for bin in ${bin_list[*]}; do
  for size in ${matrix_size[*]}; do
     bin_args=" -n ${size} -m ${wg_size}"
     echo ====================
     echo $bin
     rm -r ${prj_dir}
     advisor --collect=survey --project-dir=${prj_dir} --profile-gpu -- ${bin_dir}/${bin}${bin_args} -q
     advisor --collect=tripcounts --project-dir=${prj_dir} --flop --profile-gpu -- ${bin_dir}/${bin}${bin_args} -q
     advisor --report=roofline --gpu --project-dir=${prj_dir} --report-output=./roofline_gpu_${bin}_${device}_${size}.html -q
  done
done
    
