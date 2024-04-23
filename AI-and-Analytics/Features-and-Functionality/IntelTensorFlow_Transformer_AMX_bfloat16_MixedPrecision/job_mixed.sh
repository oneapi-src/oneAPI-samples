#!/bin/bash

echo "########## Executing the run"

source /opt/intel/oneapi/setvars.sh
source activate tensorflow

ONEDNN_VERBOSE_TIMESTAMP=1 ONEDNN_VERBOSE=1 python ./text_classification_with_transformer.py > ./logs/dnn_logs_mixed.txt

echo "########## Done with the run"
