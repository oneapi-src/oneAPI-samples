#  Copyright (c) 2022 Intel Corporation                                  
#  SPDX-License-Identifier: MIT                                          

# addresses from hls/tutorial-fpga.prj/componnet/sort_bus/sort_bus_csr.h

puts "Store to host pipes..."
master_write_32 $master_service_path 0x78 12
master_write_32 $master_service_path 0x88 99
master_write_32 $master_service_path 0x80 0xffffffff
master_write_32 $master_service_path 0x90 0xffffffff

# start component
master_write_32 $master_service_path 0x00 0x01
