## ======================================================================== ##
## Copyright 2021 Intel Corporation                                    ##
## ======================================================================== ##

import pandas as pd
from pprint import pprint
import json

expected_keys = {"total_time", "setup_time"}


def hw_config(df, column_idx):
    dict = {
        "config": df.columns[column_idx],
        "freq": int(df.iloc[1, column_idx]),
        "total_eu_count": int(df.iloc[6, column_idx]),
        "mem_bw": float(df.iloc[29, column_idx])
    }
    print("Config: ")
    pprint(dict)
    return dict


def timing_baseline(filename):
    with open(filename, 'r') as f:
        timings = json.load(f)
    for k in expected_keys:
        if k not in timings.keys():
            raise Exception("missing keys in timings data: {}".format(k))

    timings["setup_portion"] = timings['setup_time']/timings['total_time']
    print("Baseline timings: ")
    pprint(timings)
    return timings


if __name__ == '__main__':
    df = pd.read_csv('mtl_example.csv')
    timings = timing_baseline('timing.json')
    measured_arch = 2
    target_arch = 1
    mtl_spec = hw_config(df, target_arch)
    dg1_spec = hw_config(df, measured_arch)
    freq_ratio = float(mtl_spec['freq'] / dg1_spec['freq'])
    eu_ratio = float(mtl_spec['total_eu_count'] / dg1_spec['total_eu_count'])
    mem_bw_ratio = mtl_spec['mem_bw'] / dg1_spec['mem_bw']
    print(freq_ratio, eu_ratio, mem_bw_ratio)
    total_timing_projection = freq_ratio * eu_ratio * timings['total_time']
    setup_timing_projection = timings['setup_time'] / mem_bw_ratio
    print(total_timing_projection, setup_timing_projection, setup_timing_projection/total_timing_projection)
    # df.to_json('mtl_example.json')
