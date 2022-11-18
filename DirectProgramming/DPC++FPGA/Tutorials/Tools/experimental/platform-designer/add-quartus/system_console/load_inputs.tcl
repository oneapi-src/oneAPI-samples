#  Copyright (c) 2022 Intel Corporation                                  
#  SPDX-License-Identifier: MIT                                          

# addresses from add-oneapi/build/add.fpga_ip_export.prj_1/ZTSZ4mainE3Add_register_map.hpp

set ADDR_A 0x88
set VAL_A 5
set ADDR_B 0x8c
set VAL_B 3

puts "Store $VAL_A to address $ADDR_A"
master_write_32 $master_service_path $ADDR_A $VAL_A

puts "Store $VAL_B to address $ADDR_B"
master_write_32 $master_service_path $ADDR_B $VAL_B

# start component
puts "Set 'Start' bit to 1"
master_write_32 $master_service_path 0x00 0x01
