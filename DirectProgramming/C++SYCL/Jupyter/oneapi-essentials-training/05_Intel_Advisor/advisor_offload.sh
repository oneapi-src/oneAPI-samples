#!/bin/bash

advixe-python $APM/collect.py advisor_project --config gen9 -- ./matrix.dpcpp
advixe-python $APM/analyze.py advisor_project --config gen9 --out-dir ./analyze
