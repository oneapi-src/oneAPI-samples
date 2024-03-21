#!/bin/bash
advisor --collect=roofline --profile-gpu --project-dir=./../advisor/4_gpu/b8816 -- ./build/src/4_GPU_optimized 1024 1024 1024 8 8 16 100


