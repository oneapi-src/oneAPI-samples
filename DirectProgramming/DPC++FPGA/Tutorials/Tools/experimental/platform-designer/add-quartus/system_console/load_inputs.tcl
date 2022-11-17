#  Copyright (c) 2022 Intel Corporation                                  
#  SPDX-License-Identifier: MIT                                          

# addresses from hls/tutorial-fpga.prj/componnet/sort_bus/sort_bus_csr.h

puts "Store with functor..."
master_write_32 $master_service_path 0x188 5
master_write_32 $master_service_path 0x18c 16

# start component
master_write_32 $master_service_path 0x100 0x01
