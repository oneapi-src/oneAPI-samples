# launch this script from an environment containing oneAPI, Intel® Quartus®
# Prime, Platform Designer

# get the base directory of the design
set EXAMPLE_ROOT_DIR [ file dirname [ file normalize [ info script ] ] ]
set BUILD_DIR [pwd]

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
echoAndEval "file mkdir build"
echoAndEval "cd build"
post_message -type info "Detected OS = $OS"
if { $OS == "windows" } {
    echoAndExec "cmake -G \"NMake Makefiles\" $EXAMPLE_ROOT_DIR/add_oneapi"
} else {
    echoAndExec "cmake $EXAMPLE_ROOT_DIR/add_oneapi"
}
echoAndExec "cmake --build . --target report"
echoAndEval "cd $BUILD_DIR"

post_message -type info  "2. Prepare Project directory with files from starting_files"

echoAndEval "file mkdir add_quartus"
if { $OS == "windows" } {
    echoAndExec "ROBOCOPY $EXAMPLE_ROOT_DIR/starting_files/ $BUILD_DIR/add_quartus/ /S /NFL /NDL"
} else {
    set SLN_FILES [glob -dir $EXAMPLE_ROOT_DIR/starting_files *]
    echoAndExec "cp -r $SLN_FILES $BUILD_DIR/add_quartus/"
}

post_message -type info  "2.1 Create an Intel Quartus Prime project by copying the files from add_quartus_sln."
if { $OS == "windows" } {
    echoAndExec "ROBOCOPY $EXAMPLE_ROOT_DIR/add_quartus_sln/ $BUILD_DIR/add_quartus/ /S /NFL /NDL"
} else {
    set SLN_FILES [glob -dir $EXAMPLE_ROOT_DIR/add_quartus_sln *]
    echoAndExec "cp -r $SLN_FILES $BUILD_DIR/add_quartus/"
}

post_message -type info  "3. Copy the IP generated in Step 1 to the Quartus Prime project."
if { $OS == "windows" } {
    echoAndExec "ROBOCOPY $BUILD_DIR/build/add.report.prj/ $BUILD_DIR/add_quartus/add.report.prj/ /S /NFL /NDL"
} else {
    echoAndExec "cp -r $BUILD_DIR/build/add.report.prj/ $BUILD_DIR/add_quartus"
}

post_message -type info  "4. Compile Quartus Prime project"
echoAndEval "cd $BUILD_DIR/add_quartus"
echoAndExec "quartus_sh --flow compile add.qpf 2>&1 > run_quartus.log"
echoAndEval "cd $BUILD_DIR"

post_message -type info  "5. Copy the generated add.sof file to the system_console directory."
if { $OS == "windows" } {
    echoAndExec "ROBOCOPY $EXAMPLE_ROOT_DIR/system_console/ $BUILD_DIR/system_console/ /S /NFL /NDL"
    echoAndExec "xcopy add_quartus\\\\output_files\\\\add.sof system_console /Y"
} else {
    echoAndExec "cp -r $EXAMPLE_ROOT_DIR/system_console $BUILD_DIR"
    echoAndExec "cp $BUILD_DIR/add_quartus/output_files/add.sof $BUILD_DIR/system_console"
}

post_message -type info  "DONE."