import argparse
import time
import dpctl
import dpctl.tensor as dpt
import dpnp
import numpy as np


def fit_cpu(A, b, tol, max_iter):
    # Note that this function works even tensors 'A' and 'b' are NumPy or dpnp
    # arrays.
    x = np.zeros_like(b, dtype=np.float64)
    r0 = b - np.dot(A, x)
    p = r0
    for i in range(max_iter):
        a = np.inner(r0, r0) / np.inner(p, np.dot( A, p))
        x += (a * p)
        r1 = r0 - a * np.dot(A, p)
        if np.linalg.norm(r1) < tol:
            return x
        b = np.inner(r1, r1) / np.inner(r0, r0)
        p = r1 + b * p
        r0 = r1
    print('Failed to converge. Increase max-iter or tol.')
    return x

def fit(A, b, tol, max_iter):
    # Note that this function works even tensors 'A' and 'b' are NumPy or dpnp
    # arrays.
    x = dpnp.zeros_like(b, dtype=dpnp.float64)
    r0 = b - dpnp.dot(A, x)
    p = r0
    for i in range(max_iter):
        a = dpnp.inner(r0, r0) / dpnp.inner(p, dpnp.dot(A, p))
        x += a * p
        r1 = r0 - a * dpnp.dot(A, p)
        if dpnp.linalg.norm(r1) < tol:
            return x
        b = dpnp.inner(r1, r1) / dpnp.inner(r0, r0)
        p = r1 + b * p
        r0 = r1
    print('Failed to converge. Increase max-iter or tol.')
    return x


def run(gpu_id, tol, max_iter):
    """CuPy Conjugate gradient example

    Solve simultaneous linear equations, Ax = b.
    'A' and 'x' are created randomly and 'b' is computed by 'Ax' at first.
    Then, 'x' is computed from 'A' and 'b' in two ways, namely with CPU and
    GPU. To evaluate the accuracy of computation, the Euclidean distances
    between the answer 'x' and the reconstructed 'x' are computed.

    """
    for repeat in range(3):
        print('Trial: %d' % repeat)
        # Create the large symmetric matrix 'A'.
        N = 10000
        A = np.random.random((N,N))
        A = (A @ A.T).astype(np.float64)
        x_ans = np.random.random((N)).astype(np.float64)
        b = np.dot(A, x_ans)

        print('Running CPU...')
        start = time.time()
        x_cpu = fit_cpu(A, b, tol, max_iter)
        print(np.linalg.norm(x_cpu - x_ans))
        end = time.time()
        print('%s:  %f sec' % ("CPU", end - start))

        a_dpt = dpnp.asarray(A, dtype=dpnp.float64)
        b_dpt = dpnp.asarray(b, dtype=dpnp.float64)

        print('Running GPU...')
        start = time.time()
        x_gpu = fit(a_dpt, b_dpt, tol, max_iter)

        print(np.linalg.norm(dpnp.asnumpy(x_gpu) - x_ans))
        end = time.time()
        print('%s:  %f sec' % ("GPU", end - start))

        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', '-g', default=0, type=int,
                        help='ID of GPU.')
    parser.add_argument('--tol', '-t', default=0.1, type=float,
                        help='tolerance to stop iteration')
    parser.add_argument('--max-iter', '-m', default=5000, type=int,
                        help='number of iterations')
    args = parser.parse_args()
    run(args.gpu_id, args.tol, args.max_iter)
