# launch this script from an environment containing Intel® Quartus® Prime,
# Platform Designer, and the NIOS V tools (via niosv-shell).

# get the base directory of the design
set EXAMPLE_ROOT_DIR [pwd]

proc echoAndExec { exec_cmd } {
    puts "   $exec_cmd"
    eval "exec -ignorestderr $exec_cmd"
}

proc echoAndEval { tcl_cmd } {
    puts "   $tcl_cmd"
    eval "$tcl_cmd"
}

set OS [lindex $tcl_platform(platform) 0]

puts "1. Compile oneAPI kernel into an IP component"
echoAndEval "cd kernels/simple_dma"
echoAndEval "file mkdir build"
echoAndEval "cd build"
if { $OS == "Windows" } {
    echoAndExec "cmake .. -DFPGA_DEVICE=Arria10 -G \"NMake Makefiles\""
} else {
    echoAndExec "cmake .. -DFPGA_DEVICE=Arria10"
}
echoAndExec "cmake --build . --target report"
echoAndEval "cd $EXAMPLE_ROOT_DIR"

puts "2. Build sim testbench with platform designer"
echoAndExec "qsys-generate pd_system.qsys --testbench --testbench-simulation --clear-output-directory"

puts "3. Build NIOSV program"
echoAndEval "cd software/simple_dma_test"
echoAndEval "source software_build.tcl"
echoAndEval "cd $EXAMPLE_ROOT_DIR"

puts "4. Run Simulation"
echoAndEval "cd simulation_files"
echoAndExec "vsim -c -do test_nios_commandline.tcl"
set SIM_DIR [pwd]
puts ""
puts "see simulation transcript in $SIM_DIR/transcript"
puts ""
echoAndEval "cd $EXAMPLE_ROOT_DIR"

puts "DONE."