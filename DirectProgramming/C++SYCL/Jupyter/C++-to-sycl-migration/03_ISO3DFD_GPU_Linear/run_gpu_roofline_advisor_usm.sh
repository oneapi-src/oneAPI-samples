#!/bin/bash
advisor --collect=roofline --profile-gpu --project-dir=./../advisor/3_gpu/usm -- ./build/src/3_GPU_linear_USM 1024 1024 1024 100


