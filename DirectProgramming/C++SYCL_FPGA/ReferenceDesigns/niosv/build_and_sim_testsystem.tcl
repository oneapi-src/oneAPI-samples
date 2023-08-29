# launch this script from an environment containing Intel® Quartus® Prime,
# Platform Designer, and the Nios V tools (via niosv-shell).

# get the base directory of the design
set EXAMPLE_ROOT_DIR [pwd]

proc echoAndExec { exec_cmd } {
    post_message -type info  "   $exec_cmd"
    set result [catch {exec -ignorestderr -- {*}$exec_cmd 2>@1} result_text result_options]
    puts $result_text
}

proc echoAndEval { tcl_cmd } {
    post_message -type info  "   $tcl_cmd"
    eval "$tcl_cmd"
}

set OS [lindex $tcl_platform(platform) 0]

post_message -type info  "1. Compile oneAPI kernel into an IP component"
echoAndEval "cd kernels/simple_dma"
echoAndEval "file mkdir build"
echoAndEval "cd build"
post_message -type info "Detected OS = $OS"
if { $OS == "windows" } {
    echoAndExec "cmake .. -DFPGA_DEVICE=Arria10 -G \"NMake Makefiles\""
} else {
    echoAndExec "cmake .. -DFPGA_DEVICE=Arria10"
}
echoAndExec "cmake --build . --target report"
echoAndEval "cd $EXAMPLE_ROOT_DIR"

post_message -type info  "2. Build sim testbench with platform designer"
echoAndExec "qsys-generate pd_system.qsys --testbench --testbench-simulation --clear-output-directory"

post_message -type info  "3. Build Nios V program"
echoAndEval "cd software/simple_dma_test"
echoAndEval "source software_build.tcl"
echoAndEval "cd $EXAMPLE_ROOT_DIR"

post_message -type info  "4. Run Simulation"
echoAndEval "cd simulation_files"
echoAndExec "vsim -c -do test_nios_commandline.tcl"
set SIM_DIR [pwd]
post_message -type info  ""
post_message -type info  "see simulation transcript in $SIM_DIR/transcript"
post_message -type info  ""
echoAndEval "cd $EXAMPLE_ROOT_DIR"

post_message -type info  "DONE."