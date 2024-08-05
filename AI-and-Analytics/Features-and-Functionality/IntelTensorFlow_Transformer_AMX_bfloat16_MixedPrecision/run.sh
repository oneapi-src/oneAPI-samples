#!/bin/bash

echo "########## Executing the run"

source activate tensorflow

# enable verbose log
export DNNL_VERBOSE=2 
# enable JIT Dump
export DNNL_JIT_DUMP=1

DNNL_MAX_CPU_ISA=AVX512_CORE_BF16 python ./text_classification_with_transformer.py cpu >> ./logs/log_cpu_bf16_avx512_bf16.csv 2>&1

echo "########## Done with the run"
