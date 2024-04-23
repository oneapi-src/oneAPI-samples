#  Copyright (c) 2022 Intel Corporation
#  SPDX-License-Identifier: MIT

proc pause {{message "Hit Enter to continue ==> "}} {
    puts -nonewline $message
    flush stdout
    gets stdin
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

proc read_outputs {} {
    global master_service_path

    # addresses from add-oneapi/build/add.report.prj/<mangling>AdderID_register_map.hpp
    set ADDR_STATUS 0x00
    set ADDR_C 0x88
    set ADDR_FINISH_COUNT 0x30

    set readData       [master_read_32 $master_service_path $ADDR_C 2];
    set statusReg      [master_read_32 $master_service_path $ADDR_STATUS 2];
    set finishCounter  [master_read_32 $master_service_path $ADDR_FINISH_COUNT 2];

    puts "  Data   ($ADDR_C): $readData"
    puts "  Status ($ADDR_STATUS): $statusReg"
    puts "  finish ($ADDR_FINISH_COUNT): $finishCounter"
}

# set up jtag interface
get_service_paths master
set master_service_path [ lindex [get_service_paths master] 0]

open_service master $master_service_path

puts "Resetting IP..."
jtag_debug_reset_system $master_service_path

# interact with the IP
puts "TEST 1: READ OUTPUT AFTER RESET"
puts "Read outputs"
read_outputs
puts ""

puts "TEST 2: LOAD INPUTS AND CHECK OUTPUT"
pause "press 'enter' key to load inputs ==>"
load_inputs 1 2

pause "Check that IRQ LED is lit, then press 'enter' key to consume outputs ==>"
puts "Read outputs"
read_outputs
puts ""

pause "press 'enter' key to load inputs ==>"
load_inputs 3 3

pause "Check that IRQ LED is lit, then press 'enter' key to consume outputs ==>"
puts "Read outputs"
read_outputs
puts ""

puts "TEST 3: LOAD INPUTS WITHOUT CHECKING OUTPUT"
pause "press 'enter' key to load inputs ==>"
load_inputs 5 4

pause "Check that IRQ LED is lit, then press 'enter' key to overload inputs without consuming outputs ==>"
load_inputs 64 64

pause "Check that IRQ LED is lit, then press 'enter' key to overload inputs without consuming outputs ==>"
load_inputs 7 8

pause "Check that IRQ LED is lit, then press 'enter' key to consume outputs ==>"
puts "Read outputs"
read_outputs
puts ""

puts "TEST 4: READ OUTPUT AFTER NO PENDING INPUTS"
pause "press 'enter' key to consume outputs ==>"
puts "Read outputs"
read_outputs
puts ""
puts "Test complete."
