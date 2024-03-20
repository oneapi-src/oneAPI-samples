#!/bin/bash
## source /opt/intel/oneapi/embree/latest/env/vars.sh
## source /opt/intel/oneapi/openvkl/latest/env/vars.sh
## source /opt/intel/oneapi/rkcommon/latest/env/vars.sh
## source /opt/intel/oneapi/ospray/latest/env/vars.sh
## source /opt/intel/oneapi/setvars.sh --force > /dev/null
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1
/bin/echo "Running ospTutorial targeting the CPU with the command:"
/bin/echo
/bin/echo "bin/ospTutorial --osp:load-modules=cpu --osp:device=cpu"
/bin/echo
bin/ospTutorial --osp:load-modules=cpu --osp:device=cpu
