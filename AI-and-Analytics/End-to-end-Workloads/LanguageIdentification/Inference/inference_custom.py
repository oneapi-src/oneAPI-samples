#!/usr/bin/env python
# encoding: utf-8

'''
==============================================================
 Copyright © 2022 Intel Corporation
 SPDX-License-Identifier: MIT
==============================================================
'''

import os
import numpy as np
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
    def __init__(self, ipex_op=False, bf16=False, int8_model=False):
        source_model_path = "./lang_id_commonvoice_model"
        self.language_id = EncoderClassifier.from_hparams(source=source_model_path, savedir="tmp")
        print("Model: " + source_model_path)
        
        if int8_model:
            # INT8 model
            source_model_int8_path = "./lang_id_commonvoice_model_INT8"
            print("Inference with INT8 model: " + source_model_int8_path)
            from neural_compressor.utils.pytorch import load
            self.model_int8 = load(source_model_int8_path, self.language_id)
            self.model_int8.eval()
        elif ipex_op:
            # Optimize for inference with IPEX
            print("Optimizing inference with IPEX")
            self.language_id.eval()
            sampleInput = (torch.load("./sample_input_features.pt"), torch.load("./sample_input_wav_lens.pt"))
            if bf16:
                print("BF16 enabled")
                self.language_id.mods["compute_features"] = ipex.optimize(self.language_id.mods["compute_features"], dtype=torch.bfloat16)
                self.language_id.mods["mean_var_norm"] = ipex.optimize(self.language_id.mods["mean_var_norm"], dtype=torch.bfloat16)
                self.language_id.mods["embedding_model"] = ipex.optimize(self.language_id.mods["embedding_model"], dtype=torch.bfloat16)
                self.language_id.mods["classifier"] = ipex.optimize(self.language_id.mods["classifier"], dtype=torch.bfloat16)
            else:
                self.language_id.mods["compute_features"] = ipex.optimize(self.language_id.mods["compute_features"])
                self.language_id.mods["mean_var_norm"] = ipex.optimize(self.language_id.mods["mean_var_norm"])
                self.language_id.mods["embedding_model"] = ipex.optimize(self.language_id.mods["embedding_model"])
                self.language_id.mods["classifier"] = ipex.optimize(self.language_id.mods["classifier"])
            
            # Torchscript to resolve performance issues with reorder operations
            with torch.no_grad():
                I2 = self.language_id.mods["embedding_model"](*sampleInput)
                if bf16:
                    with torch.cpu.amp.autocast():
                        self.language_id.mods["compute_features"] = torch.jit.trace( self.language_id.mods["compute_features"] , example_inputs=(torch.rand(1,32000)))
                        self.language_id.mods["mean_var_norm"] = torch.jit.trace(self.language_id.mods["mean_var_norm"], example_inputs=sampleInput)
                        self.language_id.mods["embedding_model"] = torch.jit.trace(self.language_id.mods["embedding_model"], example_inputs=sampleInput)
                        self.language_id.mods["classifier"] = torch.jit.trace(self.language_id.mods["classifier"], example_inputs=I2)
                        
                        self.language_id.mods["compute_features"] = torch.jit.freeze(self.language_id.mods["compute_features"])
                        self.language_id.mods["mean_var_norm"] = torch.jit.freeze(self.language_id.mods["mean_var_norm"])
                        self.language_id.mods["embedding_model"] = torch.jit.freeze(self.language_id.mods["embedding_model"])
                        self.language_id.mods["classifier"] = torch.jit.freeze( self.language_id.mods["classifier"])
                else:
                    self.language_id.mods["compute_features"] = torch.jit.trace( self.language_id.mods["compute_features"] , example_inputs=(torch.rand(1,32000)))
                    self.language_id.mods["mean_var_norm"] = torch.jit.trace(self.language_id.mods["mean_var_norm"], example_inputs=sampleInput)
                    self.language_id.mods["embedding_model"] = torch.jit.trace(self.language_id.mods["embedding_model"], example_inputs=sampleInput)
                    self.language_id.mods["classifier"] = torch.jit.trace(self.language_id.mods["classifier"], example_inputs=I2)
                    
                    self.language_id.mods["compute_features"] = torch.jit.freeze(self.language_id.mods["compute_features"])
                    self.language_id.mods["mean_var_norm"] = torch.jit.freeze(self.language_id.mods["mean_var_norm"])
                    self.language_id.mods["embedding_model"] = torch.jit.freeze(self.language_id.mods["embedding_model"])
                    self.language_id.mods["classifier"] = torch.jit.freeze( self.language_id.mods["classifier"])

        return

    def predict(self, data_path="", ipex_op=False, bf16=False, int8_model=False, verbose=False):
        signal = self.language_id.load_audio(data_path)
        inference_start_time = time()
        
        if int8_model: # INT8 model from INC
            prediction = self.model_int8(signal)
        elif ipex_op: # IPEX
            with torch.no_grad():
                if bf16:
                    with torch.cpu.amp.autocast():
                        prediction =  self.language_id.classify_batch(signal)
                else:
                    prediction =  self.language_id.classify_batch(signal)
        else: # default
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
    parser.add_argument('-s', type=int, default=100, help="Sample size of waves to be taken from the audio file")
    parser.add_argument('--vad', action="store_true", default=False, help="Use Voice Activity Detection (VAD) to extract only the speech segments of the audio file")
    parser.add_argument('--ipex', action="store_true", default=False, help="Enable Intel Extension for PyTorch (IPEX) optimizations")
    parser.add_argument('--bf16', action="store_true", default=False, help="Use bfloat16 precision (supported on 4th Gen Xeon Scalable Processors or newer")
    parser.add_argument('--int8_model', action="store_true", default=False, help="Run inference with INT8 model generated from Intel Neural Compressor (INC)")
    parser.add_argument('--ground_truth_compare', action="store_true", default=False, help="Enable comparison of prediction labels to ground truth values")
    parser.add_argument('--verbose', action="store_true", default=False, help="Print additional debug info")
    args = parser.parse_args()

    path = args.p
    sample_dur = args.d
    sample_size = args.s
    use_vad = args.vad
    use_ipex = args.ipex
    use_bf16 = args.bf16
    use_int8_model = args.int8_model
    ground_truth_compare = args.ground_truth_compare
    verbose = args.verbose
    print("\nTaking %d samples of %d seconds each" %(sample_size, sample_dur))

    # Construct lookup table for audio files to languages
    CURR_WORKING_DIR = os.getcwd()
    if ground_truth_compare:
        AUDIO_LABELS_FILE = CURR_WORKING_DIR + "/audio_ground_truth_labels.csv"
        print("Creating lookup table from %s" %AUDIO_LABELS_FILE)
        AUDIO_LANG_LOOKUP = {}
        if os.path.exists(AUDIO_LABELS_FILE):
            with open(AUDIO_LABELS_FILE) as csv_file:
                rows = csv.reader(csv_file, delimiter=',')
                lineNum = 0
                for row in rows:
                    if 0 == lineNum:
                        # Skip column names
                        lineNum += 1
                        continue
                    else:
                        AUDIO_LANG_LOOKUP[row[0]] = row[1]
        else:
            raise Exception("Ground truth labels file does not exist.")

    speechbrain_inf = speechbrain_inference(ipex_op=use_ipex, bf16=use_bf16, int8_model=use_int8_model)
    if use_vad:
        from speechbrain.pretrained import VAD
        print("Using Voice Activity Detection")
        # Setup
        VAD_INPUT_SAMPLE_RATE = 16000 # must be 16kHz
        VAD = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="pretrained_models/vad-crdnn-libriparty")
        VAD_INPUT_FILENAME = 'input_audio_16k.wav'
        VAD_OUTPUT_FILENAME = 'vad_final.wav'
    
    if os.path.isdir(path):  
        print("Valid directory:", path)
        directory = os.fsencode(path)
    
        # CSV file for summary of output
        OUTPUT_SUMMARY_CSV_FILE = "./output_summary.csv"
        with open(OUTPUT_SUMMARY_CSV_FILE, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["Audio File", 
                             "Input Frequency", 
                             "Expected Language", 
                             "Top Consensus", 
                             "Top Consensus %", 
                             "Second Consensus", 
                             "Second Consensus %", 
                             "Average Latency", 
                             "Result"])

        total_samples = 0
        correct_predictions = 0
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if (filename.endswith(".wav") or filename.endswith(".wmv")) and 'trim_tmp.wav' != filename:
                print("\n data: %s  "%(filename))
                if ground_truth_compare:
                    if "" != AUDIO_LANG_LOOKUP[filename]:
                        print(" Expected output: %s" %AUDIO_LANG_LOOKUP[filename])
                    else:
                        print(" WARNING: no label associated. Skipping...")
                        continue
                total_samples += 1

                filepath = os.path.join(path, filename)
                if use_vad:
                    vad_start_time = time()
                    # Acquire speech segment boundaries to create an audio file with only the speech segments
                    waveform, orig_sample_rate = torchaudio.load(filepath)
                    sample_rate_for_csv = orig_sample_rate
                    # To get speech segments, the input audio file MUST be 16kHz
                    if VAD_INPUT_SAMPLE_RATE != orig_sample_rate:
                        if verbose:
                            print(" Converting audio from %dHz to %d Hz" %(orig_sample_rate, VAD_INPUT_SAMPLE_RATE))
                        transform = torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=VAD_INPUT_SAMPLE_RATE)
                        waveform = transform(waveform)
                        torchaudio.save(VAD_INPUT_FILENAME, waveform, VAD_INPUT_SAMPLE_RATE)
                        filepath = CURR_WORKING_DIR + '/' + VAD_INPUT_FILENAME
                    
                    if verbose:
                        print(" Computing speech boundaries and creating audio file with only speech segments")
                    boundaries = VAD.get_speech_segments(filepath)
                    if verbose:
                        VAD.save_boundaries(boundaries, save_path='vad_file.txt') # also prints out boundaries
                    
                    # For viewing the waveform of the time durations where speech is detected
                    #upsampled_boundaries = VAD.upsample_boundaries(boundaries, 'pretrained_model_checkpoints/' + VAD_INPUT_FILENAME)
                    #torchaudio.save('vad_output.wav', upsampled_boundaries.cpu(), VAD_INPUT_SAMPLE_RATE)

                    # Use the time boundaries to to create an audio file with only the speech segments
                    wavfile_sampleRate, wavfile_wavData = wavfile.read(filepath)
                    vad_final_wavData = []
                    for boundary in boundaries:
                        vad_final_wavData.extend(wavfile_wavData[int(boundary[0]) * wavfile_sampleRate:int(boundary[1]) * wavfile_sampleRate])
                    wavfile.write(VAD_OUTPUT_FILENAME, wavfile_sampleRate, np.asarray(vad_final_wavData))
                    filepath = os.path.join(CURR_WORKING_DIR, VAD_OUTPUT_FILENAME)
                    vad_end_time = time()
                    if verbose:
                        print("VAD processing took %.2f seconds" %(vad_end_time - vad_start_time))

                # Resample to 48kHz before loading the audio file
                waveform, orig_sample_rate = torchaudio.load(filepath)
                if not use_vad:
                    sample_rate_for_csv = orig_sample_rate
                if 48000 != orig_sample_rate:
                    RESAMPLED_AUDIO_48KHZ_FILENAME = "audio_48khz.wav"
                    if verbose:
                        print(" Resampling to 48kHz")
                    transform = torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=48000)
                    waveform = transform(waveform)
                    torchaudio.save(os.path.join(CURR_WORKING_DIR, RESAMPLED_AUDIO_48KHZ_FILENAME), waveform, 48000)
                    data = datafile(CURR_WORKING_DIR, RESAMPLED_AUDIO_48KHZ_FILENAME)
                else:
                    data = datafile(path, filename)

                # Randomly select audio segments to predict language
                predict_list = []
                use_entire_audio_file = False
                latency_sum = 0.0
                if data.waveduration < sample_dur:
                    # Use entire audio file if the duration is less than the sampling duration
                    use_entire_audio_file = True
                    sample_list = [0 for _ in range(sample_size)]
                else:
                    start_time_list = list(range(sample_size - int(data.waveduration) + 1))
                    sample_list = []
                    for i in range(sample_size):
                        sample_list.append(random.sample(start_time_list, 1)[0])
                for start in sample_list:
                    if use_entire_audio_file:
                        newWavPath = data.wavepath
                    else:
                        newWavPath = 'trim_tmp.wav'
                        data.trim_wav(newWavPath, start, start + sample_dur)
                    try:
                        label, inference_latency = speechbrain_inf.predict(data_path=newWavPath, ipex_op=use_ipex, bf16=use_bf16, int8_model=use_int8_model, verbose=verbose)
                        if verbose:
                            print(" start-end : " +  str(start)  + "  " +  str(start + sample_dur) + " prediction : " + label)
                        predict_list.append(label)
                        latency_sum += inference_latency
                    except:
                        print(" Error generating prediction")
                        predict_list.append(' ')
                        pass

                # Inference latency
                avg_latency = latency_sum / sample_size
                if verbose:
                    print(" Average latency: %.5f seconds" %(avg_latency))

                # Clean up
                if use_entire_audio_file:
                    os.remove("./" + data.filename)
                        
                # pick the top rate prediction results
                occurence_count = Counter(predict_list)
                total_count = sum(occurence_count.values())
                top_occurance = occurence_count.most_common(1)[0][0]
                top_count = occurence_count.most_common(1)[0][1]
                topPercentage = round(float(top_count/total_count)*100, 4)
                print(" Top Consensus: " + top_occurance + "  percentage: " + str(topPercentage) + "%" )
                sec_occurance = ""
                secPercentage = 0
                if topPercentage < 100.0:
                    sec_occurance = occurence_count.most_common(2)[1][0]
                    sec_count = occurence_count.most_common(2)[1][1]
                    secPercentage = round(float(sec_count/total_count)*100, 4)
                    print(" Second Consensus: " + sec_occurance + "  percentage: " + str(secPercentage) + "%" )

                if ground_truth_compare:
                    # Compare the top occurance with the expected output
                    result = "Fail"
                    if top_occurance == AUDIO_LANG_LOOKUP[filename]:
                        correct_predictions += 1
                        result = "Pass"
                    else:
                        print(" Prediction does not match top occurrance")

                    # Append result to CSV file
                    with open(OUTPUT_SUMMARY_CSV_FILE, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            filename, 
                            sample_rate_for_csv, 
                            AUDIO_LANG_LOOKUP[filename], 
                            top_occurance,
                            str(topPercentage) + "%",
                            sec_occurance, 
                            str(secPercentage) + "%", 
                            avg_latency, 
                            result
                        ])

        if ground_truth_compare:
            # Summary of results
            print("\n\n Correctly predicted %d/%d\n" %(correct_predictions, total_samples))
            print("\n See %s for summary\n" %(OUTPUT_SUMMARY_CSV_FILE))
 
    elif os.path.isfile(path):  
        print("\nIt is a normal file", path)  
    else:  
        print("It is a special file (socket, FIFO, device file)" , path)

if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
