import numpy as np
import numba_dpex
import math
from numba import cuda



@numba_dpex.kernel
def count_weighted_pairs_3d_intel_no_slm_ker(
    n,
    nbins,
    slm_hist_size,
    private_hist_size,
    x0,
    y0,
    z0,
    w0,
    x1,
    y1,
    z1,
    w1,
    rbins_squared,
    result,
):

    lid0 = numba_dpex.get_local_id(0)
    gr0 = numba_dpex.get_group_id(0)

    lid1 = numba_dpex.get_local_id(1)
    gr1 = numba_dpex.get_group_id(1)

    lws0 = numba_dpex.get_local_size(0)
    lws1 = numba_dpex.get_local_size(1)

    n_wi = 20

    dsq_mat = numba_dpex.private.array(shape=(20 * 20), dtype=np.float32)
    w0_vec = numba_dpex.private.array(shape=(20), dtype=np.float32)
    w1_vec = numba_dpex.private.array(shape=(20), dtype=np.float32)

    offset0 = gr0 * n_wi * lws0 + lid0
    offset1 = gr1 * n_wi * lws1 + lid1

    # work item works on pointer
    # j0 = gr0 * n_wi * lws0 + i0 * lws0 + lid0, and
    # j1 = gr1 * n_wi * lws1 + i1 * lws1 + lid1

    j1 = offset1
    i1 = 0
    while (i1 < n_wi) and (j1 < n):
        w1_vec[i1] = w1[j1]
        i1 += 1
        j1 += lws1

    # compute (n_wi, n_wi) matrix of squared distances in work-item
    j0 = offset0
    i0 = 0
    while (i0 < n_wi) and (j0 < n):
        x0v = x0[j0]
        y0v = y0[j0]
        z0v = z0[j0]
        w0_vec[i0] = w0[j0]

        j1 = offset1
        i1 = 0
        while (i1 < n_wi) and (j1 < n):
            dx = x0v - x1[j1]
            dy = y0v - y1[j1]
            dz = z0v - z1[j1]
            dsq_mat[i0 * n_wi + i1] = dx * dx + dy * dy + dz * dz
            i1 += 1
            j1 += lws1

        i0 += 1
        j0 += lws0

    # update slm_hist. Use work-item private buffer of 16 tfloat elements
    for k in range(0, slm_hist_size, private_hist_size):
        private_hist = numba_dpex.private.array(shape=(16), dtype=np.float32)
        for p in range(private_hist_size):
            private_hist[p] = 0.0

        j0 = offset0
        i0 = 0
        while (i0 < n_wi) and (j0 < n):
            j1 = offset1
            i1 = 0
            while (i1 < n_wi) and (j1 < n):
                dsq = dsq_mat[i0 * n_wi + i1]
                pw = w0_vec[i0] * w1_vec[i1]
                # i1 += 1
                # j1 += lws1
                pk = k
                for p in range(private_hist_size):
                    private_hist[p] += (
                        pw if (pk < nbins and dsq <= rbins_squared[pk]) else 0.0
                    )
                    pk += 1

                i1 += 1
                j1 += lws1

            i0 += 1
            j0 += lws0

        pk = k
        for p in range(private_hist_size):
            numba_dpex.atomic.add(result, pk, private_hist[p])
            pk += 1


@numba_dpex.kernel
def count_weighted_pairs_3d_intel_orig_ker(
    n, nbins, x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared, result_tmp
):
    """Naively count Npairs(<r), the total number of pairs that are separated
    by a distance less than r, for each r**2 in the input rbins_squared.
    """

    i = numba_dpex.get_global_id(0)

    px = x1[i]
    py = y1[i]
    pz = z1[i]
    pw = w1[i]
    for j in range(n):
        qx = x2[j]
        qy = y2[j]
        qz = z2[j]
        qw = w2[j]
        dx = px - qx
        dy = py - qy
        dz = pz - qz
        wprod = pw * qw
        dsq = dx * dx + dy * dy + dz * dz

        if dsq <= rbins_squared[nbins - 1]:
            for k in range(nbins - 1, -1, -1):
                if (k == 0) or (dsq > rbins_squared[k - 1]):
                    numba_dpex.atomic.add(result_tmp, k, wprod)
                    break


@numba_dpex.kernel
def count_weighted_pairs_3d_intel_agg_ker(result, result_tmp):
    i = numba_dpy.get_global_id(0)
    for j in range(i + 1):
        result[i] += result_tmp[j]


@numba_dpex.kernel
def count_weighted_pairs_3d_intel_ver1(
    n, nbins, x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared, result
):
    """Naively count Npairs(<r), the total number of pairs that are separated
    by a distance less than r, for each r**2 in the input rbins_squared.
    """

    start = numba_dpex.get_global_id(0)
    stride = numba_dpex.get_global_size(0)

    n1 = n
    n2 = n

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
                numba_dpex.atomic.add(result, k - 1, wprod)
                k = k - 1
                if k <= 0:
                    break


@numba_dpex.kernel
def count_weighted_pairs_3d_intel_ver2(
    n, nbins, x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared, result
):
    """Naively count Npairs(<r), the total number of pairs that are separated
    by a distance less than r, for each r**2 in the input rbins_squared.
    """

    i = numba_dpex.get_global_id(0)

    px = x1[i]
    py = y1[i]
    pz = z1[i]
    pw = w1[i]
    for j in range(n):
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
            numba_dpex.atomic.add(result, k - 1, wprod)
            k = k - 1
            if k <= 0:
                break