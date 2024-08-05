#!/usr/bin/env python
# encoding: utf-8

'''
==============================================================
 Copyright © 2022 Intel Corporation
 SPDX-License-Identifier: MIT
==============================================================
'''

import sys
import os
import time
import torch
import numpy as np
from neural_compressor.conf.config import Quantization_Conf
from neural_compressor.experimental import Quantization, common
from neural_compressor.utils.pytorch import load
from speechbrain.pretrained import EncoderClassifier 

DEFAULT_EVAL_DATA_PATH = "/data/commonVoice/dev"

def prepare_dataset(path):
    data_list = []
    for dir_name in os.listdir(path):
        dir_path = path + '/' + dir_name
        for file_name in os.listdir(dir_path):
            data_path = dir_path + '/' + file_name
            data_list.append((data_path, dir_name))
    return data_list

def main(argv):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=str, required=True, help="Path to the model to be optimized")
    parser.add_argument('-datapath', type=str, default=DEFAULT_EVAL_DATA_PATH, help="Path to evaluation dataset")
    args = parser.parse_args()

    model_path = args.p
    eval_data_path = args.datapath

    print("Model: " + model_path)
    language_id = EncoderClassifier.from_hparams(source=model_path, savedir="tmp")
    QUANTIZED_MODEL_PATH = model_path + "_INT8"

    class INC_DataLoader(object):
        def __init__(self):
            self.data_list = prepare_dataset(eval_data_path)
            self.batch_size = 1

        def __iter__(self):
            for data, label in self.data_list:
                try:
                    yield language_id.load_audio(path=data, savedir=os.path.join(eval_data_path, label)), label
                except:
                    # Workaround for cases where data may not load properly
                    time.sleep(2)
                    continue

    dataloader = INC_DataLoader()
    model = language_id

    def eval_func(model):
        pass_num = 0
        wrong_num = 0
        for data, label in dataloader:
            with torch.no_grad():
                prediction = model(data)
                if label == prediction: ## prediction[3][0].split(':')[0] was previous code due to changes to speechbrain.interfaces.py
                    pass_num += 1
                else:
                    wrong_num += 1
        pass_rate = pass_num / (pass_num + wrong_num)
        return pass_rate

    def benchmark(model):
        warmup = 10
        test = 100
        for data, _ in dataloader:
            break
        with torch.no_grad():
            for i in range(warmup):
                model(data)
            
            start = time.time()
            for i in range(test):
                start_test = time.time()
                model(data)
                print('inference latency:', time.time() - start_test)
        print('avg_latency:', (time.time() - start) / test)


    # Quantize model
    quant_config = Quantization_Conf()
    quant_config.usr_cfg.model.framework = "pytorch_fx"
    quant_config.usr_cfg.quantization.approach = "post_training_static_quant"
    quantizer = Quantization(quant_config)
    quantizer.model = common.Model(model)
    quantizer.calib_dataloader = dataloader
    quantizer.eval_func = eval_func
    q_model = quantizer.fit()
    q_model.save(QUANTIZED_MODEL_PATH)

    # Benchmark original model and quantized model
    print("Benchmarking original model")
    benchmark(model)
    print("Benchmarking INT8 model")
    int8_model = load(QUANTIZED_MODEL_PATH, model)
    benchmark(int8_model)

if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
