import numpy as np
import numba_dpex
import math
from numba import cuda


@cuda.jit
def count_weighted_pairs_3d_cuda_mesh(
    x1,
    y1,
    z1,
    w1,
    x2,
    y2,
    z2,
    w2,
    rbins_squared,
    result,
    ndivs,
    cell_id_indices,
    cell_id2_indices,
    num_cell2_steps,
):
    """Naive pair counting with mesh in cuda. Note x/y/z/w are
    the sorted array output by calculate_chained_mesh.
    nx, ny, nz = mesh dimensions
    """
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    nbins = rbins_squared.shape[0] - 1

    nx = ndivs[0]
    ny = ndivs[1]
    nz = ndivs[2]
    nsteps_x = num_cell2_steps[0]
    nsteps_y = num_cell2_steps[1]
    nsteps_z = num_cell2_steps[2]
    numcells = nx * ny * nz
    for icell1 in range(start, numcells, stride):
        ifirst1 = cell_id_indices[icell1]
        ilast1 = cell_id_indices[icell1 + 1]

        x_icell1 = x1[ifirst1:ilast1]
        y_icell1 = y1[ifirst1:ilast1]
        z_icell1 = z1[ifirst1:ilast1]
        w_icell1 = w1[ifirst1:ilast1]
        Ni = ilast1 - ifirst1
        if Ni > 0:
            ix1 = icell1 // (ny * nz)
            iy1 = (icell1 - ix1 * ny * nz) // nz
            iz1 = icell1 - (ix1 * ny * nz) - (iy1 * nz)

            leftmost_ix2 = max(0, ix1 - nsteps_x)
            leftmost_iy2 = max(0, iy1 - nsteps_y)
            leftmost_iz2 = max(0, iz1 - nsteps_z)

            rightmost_ix2 = min(ix1 + nsteps_x + 1, nx)
            rightmost_iy2 = min(iy1 + nsteps_y + 1, ny)
            rightmost_iz2 = min(iz1 + nsteps_z + 1, nz)

            for icell2_ix in range(leftmost_ix2, rightmost_ix2):
                for icell2_iy in range(leftmost_iy2, rightmost_iy2):
                    for icell2_iz in range(leftmost_iz2, rightmost_iz2):

                        icell2 = icell2_ix * (ny * nz) + icell2_iy * nz + icell2_iz
                        ifirst2 = cell_id2_indices[icell2]
                        ilast2 = cell_id2_indices[icell2 + 1]

                        x_icell2 = x2[ifirst2:ilast2]
                        y_icell2 = y2[ifirst2:ilast2]
                        z_icell2 = z2[ifirst2:ilast2]
                        w_icell2 = w2[ifirst2:ilast2]

                        Nj = ilast2 - ifirst2
                        if Nj > 0:
                            for i in range(0, Ni):
                                x1tmp = x_icell1[i]
                                y1tmp = y_icell1[i]
                                z1tmp = z_icell1[i]
                                w1tmp = w_icell1[i]
                                for j in range(0, Nj):
                                    # calculate the square distance
                                    dx = x1tmp - x_icell2[j]
                                    dy = y1tmp - y_icell2[j]
                                    dz = z1tmp - z_icell2[j]
                                    wprod = w1tmp * w_icell2[j]
                                    dsq = dx * dx + dy * dy + dz * dz

                                    k = nbins - 1
                                    while dsq <= rbins_squared[k]:
                                        cuda.atomic.add(result, k - 1, wprod)
                                        k = k - 1
                                        if k < 0:
                                            break


@cuda.jit
def count_weighted_pairs_3d_cuda_mesh_old(
    x1,
    y1,
    z1,
    w1,
    x2,
    y2,
    z2,
    w2,
    rbins_squared,
    result,
    ndivs,
    cell_id_indices,
    cell_id2_indices,
    num_cell2_steps,
):
    """Naive pair counting with mesh in cuda. Note x/y/z/w are
    the sorted array output by calculate_chained_mesh.
    nx, ny, nz = mesh dimensions
    """
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    nbins = rbins_squared.shape[0] - 1

    nx = ndivs[0]
    ny = ndivs[1]
    nz = ndivs[2]
    nsteps_x = num_cell2_steps[0]
    nsteps_y = num_cell2_steps[1]
    nsteps_z = num_cell2_steps[2]
    numcells = nx * ny * nz
    for icell1 in range(start, numcells, stride):
        ifirst1 = cell_id_indices[icell1]
        ilast1 = cell_id_indices[icell1 + 1]

        x_icell1 = x1[ifirst1:ilast1]
        y_icell1 = y1[ifirst1:ilast1]
        z_icell1 = z1[ifirst1:ilast1]
        w_icell1 = w1[ifirst1:ilast1]
        Ni = ilast1 - ifirst1
        if Ni > 0:
            ix1 = icell1 // (ny * nz)
            iy1 = (icell1 - ix1 * ny * nz) // nz
            iz1 = icell1 - (ix1 * ny * nz) - (iy1 * nz)

            leftmost_ix2 = max(0, ix1 - nsteps_x)
            leftmost_iy2 = max(0, iy1 - nsteps_y)
            leftmost_iz2 = max(0, iz1 - nsteps_z)

            rightmost_ix2 = min(ix1 + nsteps_x + 1, nx)
            rightmost_iy2 = min(iy1 + nsteps_y + 1, ny)
            rightmost_iz2 = min(iz1 + nsteps_z + 1, nz)

            for icell2_ix in range(leftmost_ix2, rightmost_ix2):
                for icell2_iy in range(leftmost_iy2, rightmost_iy2):
                    for icell2_iz in range(leftmost_iz2, rightmost_iz2):

                        icell2 = icell2_ix * (ny * nz) + icell2_iy * nz + icell2_iz
                        ifirst2 = cell_id2_indices[icell2]
                        ilast2 = cell_id2_indices[icell2 + 1]

                        x_icell2 = x2[ifirst2:ilast2]
                        y_icell2 = y2[ifirst2:ilast2]
                        z_icell2 = z2[ifirst2:ilast2]
                        w_icell2 = w2[ifirst2:ilast2]

                        Nj = ilast2 - ifirst2
                        if Nj > 0:
                            for i in range(0, Ni):
                                x1tmp = x_icell1[i]
                                y1tmp = y_icell1[i]
                                z1tmp = z_icell1[i]
                                w1tmp = w_icell1[i]
                                for j in range(0, Nj):
                                    # calculate the square distance
                                    dx = x1tmp - x_icell2[j]
                                    dy = y1tmp - y_icell2[j]
                                    dz = z1tmp - z_icell2[j]
                                    wprod = w1tmp * w_icell2[j]
                                    dsq = dx * dx + dy * dy + dz * dz

                                    k = nbins - 1
                                    while dsq <= rbins_squared[k]:
                                        cuda.atomic.add(result, k - 1, wprod)
                                        k = k - 1
                                        if k < 0:
                                            break


@cuda.jit
def count_weighted_pairs_3d_cuda(x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared, result):
    """Naively count Npairs(<r), the total number of pairs that are separated
    by a distance less than r, for each r**2 in the input rbins_squared.
    """
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

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
                cuda.atomic.add(result, k - 1, wprod)
                k = k - 1
                if k <= 0:
                    break


@cuda.jit
def count_weighted_pairs_3d_cuda_fix(
    x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared, result
):
    """Naively count Npairs(<r), the total number of pairs that are separated
    by a distance less than r, for each r**2 in the input rbins_squared.
    """
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    n1 = x1.shape[0]
    n2 = x2.shape[0]
    nbins = rbins_squared.shape[0]
    i = start
    while i < n1:
        px = x1[i]
        py = y1[i]
        pz = z1[i]
        pw = w1[i]
        j = 0
        while j < n2:
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
                cuda.atomic.add(result, k - 1, wprod)
                k = k - 1
                if k <= 0:
                    break
            j += 1
        i += stride


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
def count_weighted_pairs_3d_intel_diff_ker(
    n, nbins, x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared, result
):
    i = numba_dpex.get_global_id(0)

    result_pvt = numba_dpex.private.array(shape=20, dtype=np.float32)
    rbins_pvt = numba_dpex.private.array(shape=20, dtype=np.float32)

    for j in range(nbins):
        result_pvt[j] = 0.0
        rbins_pvt[j] = rbins_squared[j]

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

        if dsq <= rbins_pvt[nbins - 1]:
            for k in range(nbins - 1, -1, -1):
                if dsq > rbins_pvt[k]:
                    result_pvt[k + 1] += wprod
                    break
                if k == 0:
                    result_pvt[k] += wprod
                    break

    for j in range(nbins - 2, -1, -1):
        for k in range(j + 1, nbins, 1):
            result_pvt[k] += result_pvt[j]

    for j in range(nbins):
        result[i, j] += result_pvt[j]


@numba_dpex.kernel
def count_weighted_pairs_3d_intel_diff_agg_ker(result, n):
    col_id = numba_dpex.get_global_id(0)
    for i in range(1, n):
        result[0, col_id] += result[i, col_id]


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
    i = numba_dpex.get_global_id(0)
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