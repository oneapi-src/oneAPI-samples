#!/bin/bash

mkdir logs

wget https://raw.githubusercontent.com/IntelAI/models/master/benchmarks/common/platform_util.py

echo "########## Executing the run"

source /opt/intel/oneapi/setvars.sh
source activate tensorflow

ONEDNN_VERBOSE_TIMESTAMP=1 ONEDNN_VERBOSE=1 python ./text_classification_with_transformer.py > ./logs/dnn_logs.txt

echo "########## Done with the run"
