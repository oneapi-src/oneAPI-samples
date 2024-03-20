#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) "is running vklMinimal_CPU_01 through 06"
for i in {01..06}; do
    case="vklMinimal_CPU_${i}"
    echo
    echo "$case"
    ./bin/vklMinimal_CPU_${i}
done
