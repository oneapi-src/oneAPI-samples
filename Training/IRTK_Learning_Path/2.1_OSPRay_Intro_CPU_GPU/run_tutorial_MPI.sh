#!/bin/bash
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1
/bin/echo "Running ospTutorial with MPI with the command:"
/bin/echo
/bin/echo "mpirun -n 1 ./bin/ospTutorial --osp:load-modules=mpi --osp:device=mpiOffload \
: -n 2 ./opt/intel/oneapi/ospray/latest/bin/ospray_mpi_worker"
/bin/echo
mpirun -n 1 ./bin/ospTutorial --osp:load-modules=mpi --osp:device=mpiOffload : -n 2 ./opt/intel/oneapi/ospray/latest/bin/ospray_mpi_worker 