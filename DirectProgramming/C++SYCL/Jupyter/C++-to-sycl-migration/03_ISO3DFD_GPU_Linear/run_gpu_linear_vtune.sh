#!/bin/bash
vtune -run-pass-thru=--no-altstack -collect=gpu-offload -result-dir=vtune_dir_linear -- ./build/src/3_GPU_linear 1024 1024 1024 100
vtune -run-pass-thru=--no-altstack -collect=gpu-hotspots -result-dir=vtune_dir_hotspots_linear -- ./build/src/3_GPU_linear 1024 1024 1024 100
vtune -report summary -result-dir vtune_dir_linear -format html -report-output ./reports/output_offload_linear.html
vtune -report summary -result-dir vtune_dir_hotspots_linear -format html -report-output ./reports/output_hotspots_linear.html

