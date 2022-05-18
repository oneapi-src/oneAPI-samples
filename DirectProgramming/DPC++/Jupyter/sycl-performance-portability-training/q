#==========================================
# Copyright © 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT
#==========================================
# Script to submit job in Intel(R) DevCloud
# Version: 0.72
#==========================================

if [ -z "$1" ]; then
    echo "Missing script argument, Usage: ./q run.sh"
elif [ ! -f "$1" ]; then
    echo "File $1 does not exist"
else
    echo "Job has been submitted to Intel(R) DevCloud and will execute soon."
    echo ""
    script=$1
    property=$2
     if [ "$property" == "GPU GEN9" ]; then
             value="gen9"   
        elif [ "$property" == "GPU Iris XE Max" ]; then
            value="iris_xe_max"
        elif [ "$property" == "CPU Xeon 8153" ]; then
            value="renderkit"
        elif [ "$property" == "CPU Xeon 8256" ]; then
            value="stratix10"
        elif [ "$property" == "CPU Xeon 6128" ]; then
            value="skl"
        else
            value="gen9" 
    fi
    if [ "$property" == "{device.value}" ]; then
        echo "Selected Device is: GPU"
    else
        echo "Selected Device is: "$property
    fi
    echo ""
    # Remove old output files
    rm *.sh.* > /dev/null 2>&1
    # Submit job using qsub
    qsub_id=`qsub -l nodes=1:$value:ppn=2 -d . $script`
    job_id="$(cut -d'.' -f1 <<<"$qsub_id")"
    # Print qstat output
    qstat 
    # Wait for output file to be generated and display
    echo ""
    echo -ne "Waiting for Output "
    until [ -f $script.o$job_id ]; do
        sleep 1
        echo -ne "█"
        ((timeout++))
        # Timeout if no output file generated within 60 seconds
        if [ $timeout == 60 ]; then
            echo ""
            echo ""
            echo "TimeOut 60 seconds: Job is still queued for execution, check for output file later ($script.o$job_id)"
            echo ""
            break
        fi
    done
    # Print output and error file content if exist
    if [ -n "$(find -name '*.sh.o'$job_id)" ]; then
        echo " Done⬇"
        cat $script.o$job_id
        cat $script.e$job_id
        echo "Job Completed in $timeout seconds."
        rm *.sh.*$job_id > /dev/null 2>&1
    fi
fi
