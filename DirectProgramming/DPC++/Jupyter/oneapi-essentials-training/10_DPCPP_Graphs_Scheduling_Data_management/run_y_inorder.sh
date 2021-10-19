#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module10 -- DPCPP Graphs and dependenices - 6 of 10 y_pattern_inorder_queues.cpp
dpcpp lab/y_pattern_inorder_queues.cpp -o bin/y_pattern_inorder_queues
if [ $? -eq 0 ]; then bin/y_pattern_inorder_queues; fi

