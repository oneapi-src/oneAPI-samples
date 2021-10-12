#!/bin/bash
printf "\nsum_single_task\n"
dpcpp sum_single_task.cpp;./a.out
printf "\nsum_work_group\n"
dpcpp sum_work_group.cpp;./a.out
printf "\nsum_subgroup_reduce\n"
dpcpp sum_subgroup_reduce.cpp;./a.out
printf "\nsum_workgroup_reduce\n"
dpcpp sum_workgroup_reduce.cpp;./a.out
printf "\nsum_oneapi_reduction\n"
dpcpp sum_oneapi_reduction.cpp;./a.out
