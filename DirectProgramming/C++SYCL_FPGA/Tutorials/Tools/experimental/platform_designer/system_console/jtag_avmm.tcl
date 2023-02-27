#  Copyright (c) 2022 Intel Corporation                                  
#  SPDX-License-Identifier: MIT                                          

get_service_paths master
set master_service_path [ lindex [get_service_paths master] 0]
open_service master $master_service_path
