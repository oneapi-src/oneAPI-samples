#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
export ADVIXE_EXPERIMENTAL=gpu-profiling
advixe-cl –collect=survey --enable-gpu-profiling --project-dir=./advisor_roofline --search-dir src:r=. -- ./rtm_stencil
advixe-cl –collect=tripcounts --stacks --flop --enable-gpu-profiling --project-dir=./advisor_roofline --search-dir src:r=. -- ./rtm_stencil
advixe-cl --report=roofline --gpu --project-dir=./advisor_roofline --report-output=./advisor_roofline/roofline.html