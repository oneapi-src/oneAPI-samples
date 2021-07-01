# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2017-2021, Intel Corporation All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#     Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

env \
    LIBRARY_PATH=$PREFIX/lib/cpu_icc:$LIBRARY_PATH \
    CCL_CONFIGURATION=cpu_icc \
    LD_LIBRARY_PATH=$PREFIX/lib/cpu_icc:$LD_LIBRARY_PATH \
    CPATH=$PREFIX/include/cpu_icc:$CPATH \
    CCL_ROOT=$PREFIX \
    HOROVOD_WITHOUT_MPI=1 \
    HOROVOD_CPU_OPERATIONS=CCL \
    HOROVOD_WITHOUT_MXNET=1 \
    HOROVOD_WITHOUT_PYTORCH=1 \
    HOROVOD_WITH_TENSORFLOW=1 \
    pip install --upgrade --force-reinstall --no-cache-dir .

