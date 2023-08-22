#Usage: ./install-tools.sh T2S-compiler-path
#TODO: auto retrive libHalide.a and include files from a website
cp --remove-destination $1/Halide/lib/libHalide.a Halide/lib
cp --remove-destination $1/Halide/include/Halide.h Halide/include
cp --remove-destination $1/Halide/include/HalideBuffer.h Halide/include
cp --remove-destination $1/Halide/include/HalideRuntime.h Halide/include

cd Halide/lib
split -b 100M libHalide.a libHalide.part.
