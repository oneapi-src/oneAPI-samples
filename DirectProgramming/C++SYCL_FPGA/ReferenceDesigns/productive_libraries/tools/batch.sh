#!/bin/bash

# Usage:
#   ./batch.sh a10|s10

array=(dot     3 sdot   ddot  dsdot
       sdsdot  1 sdsdot
       dotc    2 cdotc  zdotc
       dotu    2 cdotu  zdotu
       nrm2    4 snrm2  dnrm2 scnrm2 dznrm2
       asum    4 sasum  dasum scasum dzasum
       axpy    4 saxpy  daxpy caxpy  zaxpy
       scal    4 sscal  dscal cscal  zscal
       copy    4 scopy  dcopy ccopy  zcopy
       gemm    4 sgemm  dgemm cgemm  zgemm
       symm    4 ssymm  dsymm csymm  zsymm
       hemm    2 chemm  zhemm)

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NOCOLOR='\033[0m'

path_to_tools="$( cd "$(dirname $(realpath "$BASH_SOURCE") )" >/dev/null 2>&1 ; pwd -P )" # The path to this script, which is the productive_libraries/tools directory

echo Entering productive BLAS: $path_to_tools/../blas
cd $path_to_tools/../blas

index=0
while [ "$index" -lt "${#array[*]}" ]; do
    kernel=${array[$index]}

    mkdir -p ${kernel}/build
    cd ${kernel}/build

    echo 
    echo -e ${RED}================ Testing ${kernel} =================${NOCOLOR}
    echo -e ${BLUE}Testing correctness${NOCOLOR}
    echo -e ${GREEN}Configuring ${kernel}${NOCOLOR}
    if [ "$1" = "a10" ]; then
        cmake .. >> ../../batch.out 2>&1
    else
        cmake .. -DFPGA_DEVICE=intel_s10sx_pac:pac_s10 >> ../../batch.out 2>&1
    fi

    let original_index=index
    num_variations=${array[$((index+1))]}
    let index=index+2
    for (( v=0; v<$num_variations; v++ )); do
        variation=${array[$((index))]}
        echo -e ${GREEN}Cleaning ${variation}_tiny_$1 and ${variation}_large_$1${NOCOLOR}
        make clean_${variation}_tiny_$1 >> ../../batch.out 2>&1
        make clean_${variation}_large_$1 >> ../../batch.out 2>&1
        let index=index+1
    done
    let index=original_index

    echo -e ${GREEN}Building tests of ${kernel}${NOCOLOR}
    mv ../bin/tests.sh a.sh ; rm ../bin/test* ; mv a.sh ../bin/tests.sh
    make tests  >> ../../batch.out 2>&1

    echo -e ${GREEN}Running tests of ${kernel}${NOCOLOR}
    ../bin/tests.sh

    num_variations=${array[$((index+1))]}
    let index=index+2
    for (( v=0; v<$num_variations; v++ )); do
        variation=${array[$((index))]}
      
        echo
        echo -e ${BLUE}Testing performance of ${variation}_large_$1${NOCOLOR}
        echo -e ${GREEN}Installing pre-generated files for ${variation}_large_$1${NOCOLOR}
        if make install_${variation}_large_$1 >> ../../batch.out; then
            echo -e ${GREEN}Making demo of ${variation}_large_$1${NOCOLOR}
            rm -rf  ../bin/demo_${variation}_large_$1.unsigned  ../bin/demo_${variation}_large_$1
            make demo_${variation}_large_$1 >> ../../batch.out 2>&1

            if [ "$1" = "a10" ]; then
                echo -e  ${GREEN}Unsigning the bitstream of ${variation}_large_$1${NOCOLOR}
                make unsign_${variation}_large_$1 >> ../../batch.out 2>&1

                echo -e ${GREEN}Running demo of ${variation}_large_$1${NOCOLOR}
                ../bin/demo_${variation}_large_$1.unsigned
                ret_code=$?
                if [ $ret_code -ne 0 ]; then
                    # Known issue to fix: occasional segfault. Rerun the command
                    ../bin/demo_${variation}_large_$1.unsigned
                fi
            else
                echo -e ${GREEN}Running demo of ${variation}_large_$1${NOCOLOR}
                ../bin/demo_${variation}_large_$1
                ret_code=$?
                if [ $ret_code -ne 0 ]; then
                    # Known issue to fix: occasional segfault. Rerun the command
                    ../bin/demo_${variation}_large_$1
                fi
            fi
        else
            echo -e Sorry, it seems no pre-generated files exist for ${variation}_large_$1. Skip building the demo due to the long FPGA synthesis time.
        fi
        let index=index+1
    done

    cd -
done
