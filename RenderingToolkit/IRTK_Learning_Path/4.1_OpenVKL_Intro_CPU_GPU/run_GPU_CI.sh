#!/bin/bash
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1
/bin/echo "##" $(whoami) "is running vklMinimal_GPU_01 through 06"
i=1
while [ $i -le 6 ]; do
    case=$(printf "%02d" $i)
    echo
    echo "vklMinimal_GPU_$case"
    ./bin/vklMinimal_GPU_$case
    i=$((i + 1))
done
