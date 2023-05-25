#!/usr/bin/tclsh
#  Copyright (c) 2022 Intel Corporation
#  SPDX-License-Identifier: MIT

cd add-oneapi
file mkdir build
cd build
exec cmake ..
exec cmake --build . --target report
cd ../..

file copy add-quartus-sln add-quartus

file copy add-oneapi/build/add.report.prj add-quartus
cd add-quartus
exec quartus_sh --flow compile add.qpf -c add 2>&1 > run-quartus.log