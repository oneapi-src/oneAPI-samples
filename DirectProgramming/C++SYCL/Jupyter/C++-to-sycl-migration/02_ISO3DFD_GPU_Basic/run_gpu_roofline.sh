#!/bin/bash
#advisor --collect=survey --profile-gpu --project-dir=./advi_results -- ./build/src/2_GPU_basic 256 256 256 100
#advisor --collect=tripcounts --flop --profile-gpu --project-dir=./advi_results -- ./build/src/2_GPU_basic 256 256 256 100
#advisor --collect=projection --profile-gpu --model-baseline-gpu --project-dir=./advi_results

advisor --collect=survey --profile-gpu --project-dir=./roofline -- ./build/src/2_GPU_basic 256 256 256 100
advisor --collect=tripcounts --profile-gpu --project-dir=./roofline -- ./build/src/2_GPU_basic 256 256 256 100
advisor --collect=projection --profile-gpu --model-baseline-gpu --project-dir=./roofline
advisor --report=roofline --gpu --project-dir=roofline --report-output=./roofline/roofline.html



