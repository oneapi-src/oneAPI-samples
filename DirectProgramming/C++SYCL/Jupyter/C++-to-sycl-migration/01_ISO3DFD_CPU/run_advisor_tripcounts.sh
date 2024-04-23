#!/bin/bash
advisor --collect=tripcounts --flop --no-auto-finalize --target-device=pvc_xt_448xve -- ./build/src/1_CPU_only 128 128 128 20


