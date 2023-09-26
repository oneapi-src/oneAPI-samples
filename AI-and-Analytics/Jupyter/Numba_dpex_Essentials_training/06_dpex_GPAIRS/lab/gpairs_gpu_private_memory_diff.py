
import base_gpairs_diff
import dpnp as np
import numba as nb
from numba_dpex import dpjit

# This implementation is numba dpex prange version without atomics.


@dpjit
def count_weighted_pairs_3d_diff_ker(
    n, nbins, x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared, result
):
    for i in nb.prange(n):
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
                    if dsq > rbins_squared[k]:
                        result[i, k + 1] += wprod
                        break
                    if k == 0:
                        result[i, k] += wprod
                        break

        for j in range(nbins - 2, -1, -1):
            for k in range(j + 1, nbins, 1):
                result[i, k] += result[i, j]


@dpjit
def count_weighted_pairs_3d_diff_agg_ker(nbins, result, n):
    for col_id in nb.prange(nbins):
        for i in range(1, n):
            result[0, col_id] += result[i, col_id]


def gpairs(nopt, nbins, x1, y1, z1, w1, x2, y2, z2, w2, rbins, results):
    # allocate per-work item private result vector in device global memory
    results_disjoint = np.zeros((nopt, rbins.shape[0]), dtype=np.float32)

    # call gpairs compute kernel
    count_weighted_pairs_3d_diff_ker(
        nopt, nbins, x1, y1, z1, w1, x2, y2, z2, w2, rbins, results_disjoint
    )

    # aggregate the results from the compute kernel
    count_weighted_pairs_3d_diff_agg_ker(nbins, results_disjoint, nopt)

    # copy to results vector
    results[:] = results_disjoint[0]

base_gpairs_diff.run("Gpairs Dpex kernel", gpairs)
