#!/bin/bash
## source /opt/intel/oneapi/embree/latest/env/vars.sh
## source /opt/intel/oneapi/openvkl/latest/env/vars.sh
## source /opt/intel/oneapi/rkcommon/latest/env/vars.sh
## source /opt/intel/oneapi/ospray/latest/env/vars.sh
## source /opt/intel/oneapi/setvars.sh --force > /dev/null
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1
/bin/echo "Running ospTutorial with MPI with the command:"
/bin/echo
/bin/echo "mpirun -n 1 ./bin/ospTutorial --osp:load-modules=mpi --osp:device=mpiOffload \
: -n 2 ./opt/intel/oneapi/ospray/latest/bin/ospray_mpi_worker"
/bin/echo
mpirun -n 1 ./bin/ospTutorial --osp:load-modules=mpi --osp:device=mpiOffload : -n 2 ./opt/intel/oneapi/ospray/latest/bin/ospray_mpi_worker 