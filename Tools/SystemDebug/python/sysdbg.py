#!/usr/bin/env python3
'''
==============================================================
 Copyright © 2019 Intel Corporation

 SPDX-License-Identifier: MIT
==============================================================
'''

import intel.sysdbg

intel.sysdbg.connect()

threads = intel.sysdbg.threads
target = intel.sysdbg.target

target.suspend()
threads[0].symbols.load_this()
threads[0].symbols.load_dxe()

threads[0].frames[0].symbols.find("x")[0].value().string()

threads[0].breakpoints.add(condition="Reset")

target.platform.reset()

target.resume()