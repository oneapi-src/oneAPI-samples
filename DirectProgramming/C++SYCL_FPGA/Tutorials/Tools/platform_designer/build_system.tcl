# launch this script from an environment containing oneAPI, Intel® Quartus®
# Prime, Platform Designer

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
echoAndEval "cd add_oneapi"
echoAndEval "file mkdir build"
echoAndEval "cd build"
post_message -type info "Detected OS = $OS"
if { $OS == "windows" } {
    echoAndExec "cmake .. -G \"NMake Makefiles\""
} else {
    echoAndExec "cmake .."
}
echoAndExec "cmake --build . --target report"
echoAndEval "cd $EXAMPLE_ROOT_DIR"

post_message -type info  "2. Prepare Project directory with files from starting_files"

echoAndEval "file mkdir add_quartus"
if { $OS == "windows" } {
    echoAndExec "xcopy starting_files\\\\add.sv add_quartus"
    echoAndExec "xcopy starting_files\\\\jtag.sdc add_quartus"
} else {
    echoAndExec "cp starting_files/add.sv add_quartus"
    echoAndExec "cp starting_files/jtag.sdc add_quartus"
}

post_message -type info  "2.1 Create a Intel Quartus Prime project by copying the files from add_quartus_sln."
if { $OS == "windows" } {
    echoAndExec "ROBOCOPY add_quartus_sln\\\\ add_quartus\\\\ /S /NFL /NDL"
} else {
    echoAndExec "cp -r add_quartus_sln/* add_quartus/"
}

post_message -type info  "3. Copy the IP generated in Step 1 to the Quartus Prime project."
if { $OS == "windows" } {
    echoAndExec "ROBOCOPY add_oneapi\\\\build\\\\add.report.prj\\\\ add_quartus\\\\add.report.prj\\\\ /S /NFL /NDL"
} else {
    echoAndExec "cp -r add_oneapi/build/add.report.prj/ add_quartus"
}

post_message -type info  "4. Compile Quartus Prime project"
echoAndEval "cd add_quartus"
echoAndExec "quartus_sh --flow compile add.qpf"
echoAndEval "cd $EXAMPLE_ROOT_DIR"

post_message -type info  "4. Copy the generated add.sof file to the system_console directory."
if { $OS == "windows" } {
    echoAndExec "xcopy add_quartus\\\\output_files\\\\add.sof system_console"
} else {
    echoAndExec "cp add_quartus/output_files/add.sof system_console"
}

post_message -type info  "DONE."