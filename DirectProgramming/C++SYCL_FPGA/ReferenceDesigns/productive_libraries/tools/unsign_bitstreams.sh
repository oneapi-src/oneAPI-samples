#!/bin/bash
# Usage unsign_bitstreams.sh reconfigurable array       # e.g. ssssmatmul_large_a10
#                            path_to_sign_aocx          
#                            cxx_compiler               # e.g. icpx
#                            objects                    # objects to link together with the unsigned bitstreams
reconfigurable_array=$1
path_to_sign_aocx=$2
cxx_compiler=$3
objects=$4

unsigned_bits=""
for bits in ${reconfigurable_array}*.aocx; do \
    printf 'Y\nY\nY\nY\nY\nY\n' | source ${path_to_sign_aocx}/sign_aocx.sh -H openssl_manager -i ${bits} -r NULL -k NULL -o ${bits}.unsigned
    unsigned_bits="${unsigned_bits} -fsycl-add-targets=spir64_fpga-unknown-unknown:${bits}"
done

$cxx_compiler -fsycl ${unsigned_bits} ${objects}
