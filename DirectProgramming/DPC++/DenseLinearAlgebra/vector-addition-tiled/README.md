Tiled vector addition demonstrating use of local accessors.
Note that the local accessors use a syntax that is deprecated in SYCL 2020.
Intel dpcpp surpresses warning messages.

On Intel devcloud run:
    qsub build-devcloud.sh

A script is provided for the NERSC perlmutter machine (using NVIDIA GPUs).
Currently it expects that the compiler has been build with CUDA support
and installed in $PSCRATCH/llvm-build/install (there will be perlmutter
modules with prebuilt compilers eventually). 

    bash build-perlmutter.sh

This will only build the sample. To run it you will need to allocate a GPU
node, e.g. with

    salloc -A <your account> -C gpu -q interactive -t 10:00 -n 1

Then you can run the binary with:

    ONEAPI_DEVICE_SELECTOR=cuda:0 ./vector-addition-tiled

It should be easy to modify the script local installations as long as you have a
version of clang++ that is build for SYCL/CUDA.

Similar methods should work for SYCL/HIP.
