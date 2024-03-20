#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) "is running vklMinimal_GPU_01 through 06"
for i in {01..06}; do
    case="vklMinimal_GPU_${i}"
    echo
    echo
    echo "$case"
    ./bin/vklMinimal_GPU_${i}
done
