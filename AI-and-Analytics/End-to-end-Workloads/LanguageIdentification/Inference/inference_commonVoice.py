#!/usr/bin/env python
# encoding: utf-8

'''
==============================================================
 Copyright © 2022 Intel Corporation
 SPDX-License-Identifier: MIT
==============================================================
'''

import os
import random
import csv
from time import time
from collections import Counter
from scipy.io import wavfile

import torch
import torchaudio
import intel_extension_for_pytorch as ipex
from speechbrain.pretrained import EncoderClassifier

class datafile:
    def __init__(self, dirpath, filename):
        self.dirpath = dirpath
        self.filename = filename
        self.wavefile = ''
        self.wavepath = ''
        self.sampleRate = 0
        self.waveData = ''
        self.wavesize = 0
        self.waveduriation = 0
        if filename.endswith(".wav") or filename.endswith(".wmv"):
            self.wavefile = filename
            self.wavepath = dirpath + os.sep + filename
            self.sampleRate, self.waveData = wavfile.read( self.wavepath )
            self.wavesize = self.waveData.size
            self.waveduration = self.wavesize / self.sampleRate
            self.wavelength = self.waveData.shape[0] / self.sampleRate
            print(" wave dur, len, size, rate : ", self.waveduration, self.wavelength , self.wavesize , self.sampleRate)
        return

    def trim_wav(self, newWavPath , start, end ):
        startSample = int( start * self.sampleRate )
        endSample = int( end * self.sampleRate )
        wavfile.write( newWavPath, self.sampleRate, self.waveData[startSample:endSample])

class speechbrain_inference:
    def __init__(self, ipex_op=False):
        source_model_path = "./lang_id_commonvoice_model"
        self.language_id = EncoderClassifier.from_hparams(source=source_model_path, savedir="tmp")
        print("Model: " + source_model_path)
        
        # Optimize for inference with IPEX
        if ipex_op:
            print("Optimizing inference with IPEX")
            self.language_id.eval()
            sampleInput = (torch.load("./sample_input_features.pt"), torch.load("./sample_input_wav_lens.pt"))
            self.language_id.mods["embedding_model"] = ipex.optimize(self.language_id.mods["embedding_model"], sample_input=sampleInput)
            # Torchscript to resolve performance issues with reorder operations
            self.language_id.mods["embedding_model"] = torch.jit.trace(self.language_id.mods["embedding_model"], example_inputs=sampleInput)
        return

    def predict(self, data_path="", verbose=False):
        signal = self.language_id.load_audio(data_path)
        inference_start_time = time()
        prediction =  self.language_id.classify_batch(signal)
        inference_end_time = time()
        inference_latency = inference_end_time - inference_start_time
        if verbose:
            print(" Inference latency: %.5f seconds" %(inference_latency))
        
        # prediction is a tuple of format (out_prob, score, index) due to modification of speechbrain.pretrained.interfaces.py
        label = self.language_id.hparams.label_encoder.decode_torch(prediction[2])[0]
        
        return label, inference_latency

def main(argv):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=str, required=True, help="Path to the audio data files for inference")
    parser.add_argument('-d', type=int, default=3, help="Duration of each wave sample in seconds")
    parser.add_argument('-s', type=int, default=5, help="Sample size of waves to be taken from the audio file")
    parser.add_argument('--ipex', action="store_true", default=False, help="Enable Intel Extension for PyTorch (IPEX) optimizations")
    parser.add_argument('--verbose', action="store_true", default=False, help="Print additional debug info")
    args = parser.parse_args()

    path = args.p
    sample_dur = args.d
    sample_size = args.s
    use_ipex = args.ipex
    verbose = args.verbose
    print("\nTaking %d samples of %d seconds each" %(sample_size, sample_dur))
    
    # Acquire list of all available languages
    CURR_WORKING_DIR = os.getcwd()
    languageList = os.listdir(path)
    languageList.sort()

    # Set up output .csv file
    RESULTS_CSV_FILE = "./test_data_accuracy.csv"
    with open(RESULTS_CSV_FILE, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Language", "Total Samples", "Correct Predictions", "Accuracy"])

    speechbrain_inf = speechbrain_inference(ipex_op=use_ipex)
    for language in languageList:
        print("\nTesting on %s data" %language)
        testDataDirectory = path + "/" + language
        num_samples = len(os.listdir(testDataDirectory))
        total_correct = 0
        for file in os.listdir(testDataDirectory):
            filename = os.fsdecode(file)
            if (filename.endswith(".wav") or filename.endswith(".wmv")) and 'trim_tmp.wav' != filename:
                # Perform inference with the pretrained model by randomly selecting audio segments to predict language
                data = datafile(testDataDirectory, filename)
                predict_list = []
                use_entire_audio_file = False
                if data.waveduration < sample_dur:
                    # Use entire audio file if the duration is less than the sampling duration
                    use_entire_audio_file = True
                    sample_list = [0 for _ in range(sample_size)]
                else:
                    sample_list = [random.randint(sample_dur, int(data.waveduration)) - sample_dur for _ in range(sample_size)]
                for start in sample_list:
                    if use_entire_audio_file:
                        newWavPath = data.wavepath
                    else:
                        newWavPath = 'trim_tmp.wav'
                        data.trim_wav(newWavPath, start, start + sample_dur)
                    try:
                        label, inference_latency = speechbrain_inf.predict(data_path=newWavPath, verbose=verbose)
                        if verbose:
                            print(" start-end : " +  str(start)  + "  " +  str(start + sample_dur) + " prediction : " + label)
                        predict_list.append(label)
                    except:
                        print("Error generating prediction")
                        predict_list.append(' ')
                        pass

                # Clean up
                if use_entire_audio_file:
                    os.remove("./" + data.filename)

                # Pick the top rated prediction result
                occurence_count = Counter(predict_list)
                total_count = sum(occurence_count.values())
                top_occurance = occurence_count.most_common(1)[0][0]
                top_count = occurence_count.most_common(1)[0][1]
                topPercentage = round(float(top_count/total_count)*100, 4)
                print(" Top Consensus: " + top_occurance + "  percentage: " + str(topPercentage) + "%" )

                # Compare the top occurance with the expected output
                if top_occurance == language:
                    total_correct = total_correct + 1

        # Compute the accuracy and write it to the csv file
        accuracy = float(total_correct) / num_samples
        print("Accuracy: %.2f%%" %(accuracy*100))
        with open(RESULTS_CSV_FILE, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([
                language,
                num_samples,
                total_correct,
                str(accuracy*100) + "%"
            ])
            
    print("\n See %s for summary" %(RESULTS_CSV_FILE))


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
