# *****************************************************************************
# Copyright (c) 2020, Intel Corporation All rights reserved.
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

import numpy as np


def __gpairs_ref__(x1, y1, z1, w1, x2, y2, z2, w2, rbins):
    dm = (
        np.square(x2 - x1[:, None])
        + np.square(y2 - y1[:, None])
        + np.square(z2 - z1[:, None])
    )
    return np.array(
        [np.outer(w1, w2)[dm <= rbins[k]].sum() for k in range(len(rbins))]
    )


def gpairs_python(x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared, result):
    result[:] = __gpairs_ref__(x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared)

    #### Old implementation #####
    # n1 = x1.shape[0]
    # n2 = x2.shape[0]
    # nbins = rbins_squared.shape[0]

    # for i in range(n1):
    #     px = x1[i]
    #     py = y1[i]
    #     pz = z1[i]
    #     pw = w1[i]
    #     for j in range(n2):
    #         qx = x2[j]
    #         qy = y2[j]
    #         qz = z2[j]
    #         qw = w2[j]
    #         dx = px - qx
    #         dy = py - qy
    #         dz = pz - qz
    #         wprod = pw * qw
    #         dsq = dx * dx + dy * dy + dz * dz

    #         k = nbins - 1
    #         while dsq <= rbins_squared[k]:
    #             result[k - 1] += wprod
    #             k = k - 1
    #             if k <= 0:
    #                 break