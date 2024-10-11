#!/bin/sh
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1
/bin/echo "##" $(whoami) "is running vklMinimal_CPU_01 through 06"
i=1
while [ $i -le 6 ]; do
    case=$(printf "%02d" $i)
    echo
    echo "vklMinimal_CPU_$case"
    ./bin/vklMinimal_CPU_$case
    i=$((i + 1))
done
