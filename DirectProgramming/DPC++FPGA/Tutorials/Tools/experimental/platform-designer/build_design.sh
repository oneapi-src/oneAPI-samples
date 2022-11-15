#!/bin/bash -x
cd add-oneapi &&
mkdir build &&
cd build &&
cmake .. &&
make fpga_ip_export &&
cd ../../ &&

cp -r add-oneapi/build/add.fpga_ip_export.prj_1/ add-quartus/ &&
cp -r add-oneapi/build/add.fpga_ip_export.prj_5/ add-quartus/ &&

cd add-quartus &&

cd add.fpga_ip_export.prj_1 &&
python add_fpga_ip_export_1_di_hw_tcl_adjustment_script.py  &&

cd .. &&
cd add.fpga_ip_export.prj_5 &&
python add_fpga_ip_export_5_di_hw_tcl_adjustment_script.py &&

cd .. &&

qsys-script --script=add_kernels.tcl --quartus-project=AddCSRDemo.qpf &&
# qsys-generate add_kernels.qsys --synthesis --quartus-project=AddCSRDemo.qpf --rev=add &&
quartus_sh --flow compile AddCSRDemo.qpf -c add &&
echo "DONE build_designs.sh"