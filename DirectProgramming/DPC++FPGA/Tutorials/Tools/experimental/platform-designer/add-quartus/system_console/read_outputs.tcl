#  Copyright (c) 2022 Intel Corporation                                  
#  SPDX-License-Identifier: MIT                                          

# addresses from add-oneapi/build/add.fpga_ip_export.prj_1/ZTSZ4mainE3Add_register_map.hpp
set ADDR_STATUS 0x00
set ADDR_C 0x78
set ADDR_FINISH_COUNT 0x28


puts "Outputs:"

set readData       [master_read_32 $master_service_path $ADDR_C 2];
set statusReg      [master_read_32 $master_service_path $ADDR_STATUS 1];
set finishCounter  [master_read_32 $master_service_path $ADDR_FINISH_COUNT 2];

puts "  Data ($ADDR_C): $readData"
puts "  Status ($ADDR_STATUS): $statusReg"
puts "  finish ($ADDR_FINISH_COUNT): $finishCounter"