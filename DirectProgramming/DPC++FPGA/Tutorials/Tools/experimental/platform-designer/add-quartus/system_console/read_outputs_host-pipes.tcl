#  Copyright (c) 2022 Intel Corporation                                  
#  SPDX-License-Identifier: MIT                                          

# addresses from hls/tutorial-fpga.prj/componnet/sort_bus/sort_bus_csr.h

# clear interrupt flag
# master_write_32 $master_service_path 0x98 1


puts "host-pipes input:"

set readData       [master_read_32 $master_service_path 0x98 2];
set readDataValid  [master_read_32 $master_service_path 0xa0 2];
set statusReg      [master_read_32 $master_service_path 0x00 1];
set finishCounter0 [master_read_32 $master_service_path 0x28 1];
set finishCounter1 [master_read_32 $master_service_path 0x2c 1];

puts "  Data: $readData"
puts "  Valid: $readDataValid"
puts "  status: $statusReg"
puts "  finish0 $finishCounter0"
puts "  finish1 $finishCounter1"