
# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import base_gpairs_gpu
import numpy as np
import gwpc_private as gwpc
import dpctl, dpctl.tensor as dpt
from device_selector import get_device_selector
import dpctl
from numba_dpex import kernel, atomic, DEFAULT_LOCAL_SIZE
import numba_dpex

atomic_add = atomic.add


@kernel
def count_weighted_pairs_3d_intel(
    x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared, result
):
    """Naively count Npairs(<r), the total number of pairs that are separated
    by a distance less than r, for each r**2 in the input rbins_squared.
    """

    start = numba_dpex.get_global_id(0)
    stride = numba_dpex.get_global_size(0)

    n1 = x1.shape[0]
    n2 = x2.shape[0]
    nbins = rbins_squared.shape[0]

    for i in range(start, n1, stride):
        px = x1[i]
        py = y1[i]
        pz = z1[i]
        pw = w1[i]
        for j in range(n2):
            qx = x2[j]
            qy = y2[j]
            qz = z2[j]
            qw = w2[j]
            dx = px - qx
            dy = py - qy
            dz = pz - qz
            wprod = pw * qw
            dsq = dx * dx + dy * dy + dz * dz

            k = nbins - 1
            while dsq <= rbins_squared[k]:
                atomic_add(result, k - 1, wprod)
                k = k - 1
                if k <= 0:
                    break


@kernel
def count_weighted_pairs_3d_intel_ver2(
    x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared, result
):
    """Naively count Npairs(<r), the total number of pairs that are separated
    by a distance less than r, for each r**2 in the input rbins_squared.
    """

    i = numba_dpex.get_global_id(0)
    nbins = rbins_squared.shape[0]
    n2 = x2.shape[0]

    px = x1[i]
    py = y1[i]
    pz = z1[i]
    pw = w1[i]
    for j in range(n2):
        qx = x2[j]
        qy = y2[j]
        qz = z2[j]
        qw = w2[j]
        dx = px - qx
        dy = py - qy
        dz = pz - qz
        wprod = pw * qw
        dsq = dx * dx + dy * dy + dz * dz

        k = nbins - 1
        while dsq <= rbins_squared[k]:
            # disabled for now since it's not supported currently
            # - could reenable later when it's supported (~April 2020)
            # - could work around this to avoid atomics, which would perform better anyway
            # cuda.atomic.add(result, k-1, wprod)
            atomic_add(result, k - 1, wprod)
            k = k - 1
            if k <= 0:
                break


def ceiling_quotient(n, m):
    return int((n + m - 1) / m)


def count_weighted_pairs_3d_intel_no_slm(
    n, nbins, d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2, d_rbins_squared, d_result
):
    n_wi = 20
    private_hist_size = 16
    lws0 = 16
    lws1 = 16

    m0 = n_wi * lws0
    m1 = n_wi * lws1

    n_groups0 = ceiling_quotient(n, m0)
    n_groups1 = ceiling_quotient(n, m1)

    gwsRange = n_groups0 * lws0, n_groups1 * lws1
    lwsRange = lws0, lws1

    slm_hist_size = ceiling_quotient(nbins, private_hist_size) * private_hist_size

    with dpctl.device_context(base_gpairs_gpu.get_device_selector(is_gpu=True)):
        gwpc.count_weighted_pairs_3d_intel_no_slm_ker[gwsRange, lwsRange](
            n,
            nbins,
            slm_hist_size,
            private_hist_size,
            d_x1,
            d_y1,
            d_z1,
            d_w1,
            d_x2,
            d_y2,
            d_z2,
            d_w2,
            d_rbins_squared,
            d_result,
        )


def count_weighted_pairs_3d_intel_orig(
    n, nbins, d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2, d_rbins_squared, d_result
):

    # create tmp result on device
    result_tmp = np.zeros(nbins, dtype=np.float32)
    d_result_tmp = dpt.usm_ndarray(
        result_tmp.shape, dtype=result_tmp.dtype, buffer="device"
    )
    d_result_tmp.usm_data.copy_from_host(result_tmp.reshape((-1)).view("|u1"))

    with dpctl.device_context(base_gpairs_gpu.get_device_selector()):
        gwpc.count_weighted_pairs_3d_intel_orig_ker[n,](
            n,
            nbins,
            d_x1,
            d_y1,
            d_z1,
            d_w1,
            d_x2,
            d_y2,
            d_z2,
            d_w2,
            d_rbins_squared,
            d_result_tmp,
        )
        gwpc.count_weighted_pairs_3d_intel_agg_ker[
            nbins,
        ](d_result, d_result_tmp)


def run_gpairs(
    n, nbins, d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2, d_rbins_squared, d_result
):
    count_weighted_pairs_3d_intel_no_slm(
        n,
        nbins,
        d_x1,
        d_y1,
        d_z1,
        d_w1,
        d_x2,
        d_y2,
        d_z2,
        d_w2,
        d_rbins_squared,
        d_result,
    )


base_gpairs_gpu.run("Gpairs Dpex kernel", run_gpairs)
