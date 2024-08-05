
# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import base_gpairs_gpu_graph
import numpy as np
import gaussian_weighted_pair_counts_gpu as gwpc
import numba_dpex
import dpctl


def run_gpairs(
    d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2, d_rbins_squared, d_result
):
    blocks = 512

    with dpctl.device_context(base_gpairs_gpu_graph.get_device_selector(is_gpu=True)):
        gwpc.count_weighted_pairs_3d_intel_ver2[
            d_x1.shape[0], numba_dpex.DEFAULT_LOCAL_SIZE
        ](d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2, d_rbins_squared, d_result)


base_gpairs_gpu_graph.run("Gpairs Dpex kernel", run_gpairs)
