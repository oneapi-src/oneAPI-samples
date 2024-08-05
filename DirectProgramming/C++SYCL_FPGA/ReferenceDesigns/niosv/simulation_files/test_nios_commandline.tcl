set QSYS_SIMDIR ../pd_system_tb/pd_system_tb/sim
source ../pd_system_tb/pd_system_tb/sim/mentor/msim_setup.tcl
ld_debug
source wave.do
run 6ms
exit