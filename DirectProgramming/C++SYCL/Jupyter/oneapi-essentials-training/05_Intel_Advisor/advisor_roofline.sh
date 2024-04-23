#!/bin/bash

advixe-cl –collect=survey --enable-gpu-profiling --project-dir=./adv -- ./matrix.dpcpp

advixe-cl -–collect=tripcounts --stacks --flop --enable-gpu-profiling --project-dir=./adv -- ./matrix.dpcpp

advixe-cl --report=roofline --gpu  --project-dir=./adv

