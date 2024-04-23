#!/bin/bash
vtune -run-pass-thru=--no-altstack -collect=gpu-offload -result-dir=vtune_dir -- ./build/src/2_GPU_basic 1024 1024 1024 100
vtune -run-pass-thru=--no-altstack -collect=gpu-hotspots -result-dir=vtune_dir_hotspots -- ./build/src/2_GPU_basic 1024 1024 1024 100
vtune -report summary -result-dir vtune_dir -format html -report-output ./reports/output_offload.html
vtune -report summary -result-dir vtune_dir_hotspots -format html -report-output ./reports/output_hotspots.html

