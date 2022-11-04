#  Copyright (c) 2022 Intel Corporation                                  
#  SPDX-License-Identifier: MIT                                          

# addresses from hls/tutorial-fpga.prj/componnet/sort_bus/sort_bus_csr.h

# clear interrupt flag
# master_write_32 $master_service_path 0x98 1

puts "functor members input:"

set readData       [master_read_32 $master_service_path 0x178 2];
set readDataValid  [master_read_32 $master_service_path 0x180 2];
set statusReg      [master_read_32 $master_service_path 0x100 1];
set finishCounter0 [master_read_32 $master_service_path 0x128 1];
set finishCounter1 [master_read_32 $master_service_path 0x12c 1];

puts "  Data: $readData"
puts "  Valid: $readDataValid"
puts "  status: $statusReg"
puts "  finish0 $finishCounter0"
puts "  finish1 $finishCounter1"