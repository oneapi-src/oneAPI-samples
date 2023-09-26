
import base_gpairs_naive
import gaussian_weighted_pair_counts_gpu as gwpc


def ceiling_quotient(n, m):
    return int((n + m - 1) / m)

def count_weighted_pairs_3d_intel_no_slm(
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

    slm_hist_size = (
        ceiling_quotient(nbins, private_hist_size) * private_hist_size
    )

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


def run_gpairs(
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


base_gpairs_naive.run("Gpairs Dpex kernel", run_gpairs)
