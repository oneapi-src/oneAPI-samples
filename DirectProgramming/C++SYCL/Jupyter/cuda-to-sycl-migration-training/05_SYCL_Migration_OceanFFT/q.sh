#!/bin/bash
#==========================================
# Copyright © Intel Corporation
#
# SPDX-License-Identifier: MIT
#==========================================
# Script to submit job in Intel(R) DevCloud
#
#  Usage   : ./q.sh <script> <optional: node_name>
#  Example : ./q.sh run.sh gen9
#  *ignores DevCloud stuff if run on local system
#   
# Version: 0.8
#==========================================
if [ -z "$1" ]; then
    echo "Missing script argument, Usage: ./q.sh run.sh"
    echo "Missing script argument, Usage: ./q.sh run.sh gen9"
elif [ ! -f "$1" ]; then
    echo "File $1 does not exist"
else
    # If running on DevCloud, uses qsub to submit to a gpu node
    if [ -x "$(command -v qsub)" ]; then
        echo "Job has been submitted to Intel(R) DevCloud and will execute soon."
        echo ""
        # If node is specified use it, if not gpu node is default
        script=$1
        device=gpu
        if [ ! -z "$2" ]; then
            device=$2
        fi
        echo "Executing on \"$device\" node"
        echo ""
        # Remove old output files
        rm *.sh.* > /dev/null 2>&1
        # Submit job using qsub
        qsub_id=`qsub -l nodes=1:$device:ppn=2 -d . $script`
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
            if [ $timeout == 70 ]; then
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
    else
        # If not running on DevCloud, just run the script, igores the rest
        ./$1;
    fi
fi
