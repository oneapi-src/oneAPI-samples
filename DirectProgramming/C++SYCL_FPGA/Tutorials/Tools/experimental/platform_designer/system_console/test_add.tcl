#  Copyright (c) 2022 Intel Corporation
#  SPDX-License-Identifier: MIT

proc setup_jtag {} {
    global master_service_path

    get_service_paths master
    set master_service_path [ lindex [get_service_paths master] 0]
    open_service master $master_service_path

    puts "reset component"
    jtag_debug_sample_reset
}

proc load_inputs { VAL_A VAL_B } {
    global master_service_path

    # addresses from add-oneapi/build/add.report.prj/<mangling>AdderID_register_map.hpp
    set ADDR_A 0x80
    set ADDR_B 0x84
    set ADDR_START 0x08

    puts "Store $VAL_A to address $ADDR_A"
    master_write_32 $master_service_path $ADDR_A $VAL_A

    puts "Store $VAL_B to address $ADDR_B"
    master_write_32 $master_service_path $ADDR_B $VAL_B

    # start component
    puts "Set 'Start' bit to 1"
    master_write_32 $master_service_path $ADDR_START 0x01
}

proc read_outputs_no_confirm {} {
    global master_service_path

    # addresses from add-oneapi/build/add.report.prj/<mangling>AdderID_register_map.hpp
    set ADDR_STATUS 0x00
    set ADDR_C 0x88
    set ADDR_C_VALID 0x98
    set ADDR_C_READY 0x90
    set ADDR_FINISH_COUNT 0x30

    puts "Outputs:"

    set readData       [master_read_32 $master_service_path $ADDR_C 2];
    set readDataValid  [master_read_32 $master_service_path $ADDR_C_VALID 2];
    set readDataReady  [master_read_32 $master_service_path $ADDR_C_READY 2];
    set statusReg      [master_read_32 $master_service_path $ADDR_STATUS 2];
    set finishCounter  [master_read_32 $master_service_path $ADDR_FINISH_COUNT 2];

    puts "  Data ($ADDR_C): $readData"
    puts "  Data Ready ($ADDR_C_READY): $readData"
    puts "  Data Valid ($ADDR_C_VALID): $readDataValid"
    puts "  Status ($ADDR_STATUS): $statusReg"
    puts "  finish ($ADDR_FINISH_COUNT): $finishCounter"
}

proc read_outputs { } {
    global master_service_path

    # addresses from add-oneapi/build/add.report.prj/<mangling>AdderID_register_map.hpp
    set ADDR_C_READY 0x90

    # read the output registers
    read_outputs_no_confirm

    # set the `ready` bit to indicate that data has been consumed
    set readDataReady 1;
    puts "Store $readDataReady to address $ADDR_C_READY"
    master_write_32 $master_service_path $ADDR_C_READY $readDataReady
}

# Example run
setup_jtag
load_inputs 4 5
read_outputs_no_confirm
load_inputs 3 4
read_outputs_no_confirm
load_inputs 1 2
read_outputs
