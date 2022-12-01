#!/bin/bash -x
cd add-oneapi &&
mkdir build &&
cd build &&
cmake .. &&
make fpga_ip_export &&
cd ../../ &&

cp -r add-oneapi/build/add.fpga_ip_export.prj_1/ add-quartus/ &&

cd add-quartus &&

cd add.fpga_ip_export.prj_1 &&
python add_fpga_ip_export_1_di_hw_tcl_adjustment_script.py  &&

cd .. &&

qsys-script --script=add_kernel_wrapper.tcl --quartus-project=add.qpf 2>&1 > platform-designer.log &&
quartus_sh --flow compile add.qpf -c add 2>&1 > quartus.log &&
echo "DONE build_designs.sh"