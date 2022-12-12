Vector addition showing different ways to express conditionals.

On Intel devcloud run:
    qsub build-devcloud.sh

A script is provided for the NERSC perlmutter machine (using NVIDIA GPUs),
you can use:
    bash build-perlmutter.sh

It should be easy to modify for local installations as long as you have a
version of clang++ that is build for SYCL/CUDA.

Similar methods should work for SYCL/HIP.
