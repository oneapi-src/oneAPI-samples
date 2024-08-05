#!/bin/bash
#advisor --collect=roofline --profile-gpu --project-dir=./../advisor/3_gpu -- ./build/src/3_GPU_linear 256 256 256 100

advisor --collect=survey --profile-gpu -project-dir=./roofline_linear -- ./build/src/4_GPU_optimized 1024 1024 1024 32 8 4 100
advisor --collect=tripcounts --profile-gpu --project-dir=./roofline_linear -- ./build/src/4_GPU_optimized 1024 1024 1024 32 8 4 100
advisor --collect=projection --profile-gpu --model-baseline-gpu --project-dir=./roofline_linear
advisor --report=roofline --gpu --project-dir=roofline_linear --report-output=./roofline_linear/roofline_linear.html



