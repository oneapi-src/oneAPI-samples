#!/bin/bash
# Usage: ./pre_gen.sh VARIATION_SIZE_HW
# Here VARIATION is a kernel's variation, SIZE is tiny or large, and HW is a10 or s10.

# Bash version must be >= 4
bash_version=$BASH_VERSINFO
echo Bash version: $bash_version
if (($bash_version<4)); then
    echo "Error: Bash version >= 4.0 expected to run the script"
    exit
fi

path_to_tools="$( cd "$(dirname $(realpath "$BASH_SOURCE") )" >/dev/null 2>&1 ; pwd -P )" # The path to this script, which is the productive_libraries/tools directory

echo Entering productive BLAS: $path_to_tools/../blas
cd $path_to_tools/../blas

source pre_generated/info.sh

if test "${kernel_to_tarball[$1]+exists}"; then
    if test "${kernel_to_reconfigurable_array[$1]+exists}"; then
        echo Creating pre_generated/$tarball ...
        tarball=${kernel_to_tarball[$1]}
        array=${kernel_to_reconfigurable_array[$1]}
        variation=$(echo $tarball | cut -d'_' -f1)
        size=$(echo $tarball | cut -d'_' -f2)
        hw=$(echo $tarball | cut -d'_' -f3)
        rm -rf pre_generated/$tarball
        tar czvf pre_generated/$tarball $array/bin/${variation}_${size}_${hw}.a $array/oneapi/${variation}_${size}_${hw}.cpp  $array/reports/${variation}_${size}_${hw}
    fi
else
    echo "Error: No enry in dictionary kernel_to_tarball or kernel_to_reconfigurable_array for $1"
    exit 1
fi

cd -
