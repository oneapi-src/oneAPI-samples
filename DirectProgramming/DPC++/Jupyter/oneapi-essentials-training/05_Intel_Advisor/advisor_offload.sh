#!/bin/bash
rm -rf advisor_offload
advixe-python $APM/collect.py advisor_offload --config gen9 -- ./a.out
advixe-python $APM/analyze.py advisor_offload --config gen9 --out-dir ./advisor_offload/report