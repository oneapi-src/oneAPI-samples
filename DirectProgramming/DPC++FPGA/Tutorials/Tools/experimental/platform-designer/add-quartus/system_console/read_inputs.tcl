#  Copyright (c) 2022 Intel Corporation                                  
#  SPDX-License-Identifier: MIT                                          

# addresses from hls/tutorial-fpga.prj/componnet/sort_bus/sort_bus_csr.h

puts "host-pipes input"

set a_val [master_read_32 $master_service_path 0x78 2];
set b_val [master_read_32 $master_service_path 0x88 2];

puts "  a = $a_val"
puts "  b = $b_val"

puts "functor members input"

set a_val [master_read_32 $master_service_path 0x88 1];
set b_val [master_read_32 $master_service_path 0x8c 1];

puts "  a = $a_val"
puts "  b = $b_val"

# master_read_memory $master_service_path 0x00 256