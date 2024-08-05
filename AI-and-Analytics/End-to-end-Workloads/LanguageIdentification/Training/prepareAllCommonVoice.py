#!/usr/bin/env python
# encoding: utf-8

'''
==============================================================
 Copyright © 2022 Intel Corporation
 SPDX-License-Identifier: MIT
==============================================================
'''

import os
import shutil
import argparse
import csv
import random
from pydub import AudioSegment

# Modify list of languages based on application
LANGUAGE_PATHS = [
        "/japanese/ja",
        "/swedish/sv-SE",   
    ]

# Generates a CSV file for one use case: training, validation, or testing
def createCsvFile(audio_clips, start, end, output_path):
    print("Output: %s" %output_path)
    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        for i in range(start,end):
            writer.writerow([audio_clips[i]])
    return

# Generates CSV files for training (80%), validation (10%), and testing (10%)
def createAllCsvFiles(common_voice_path, maxSamples):
    for language_path in LANGUAGE_PATHS:
        save_folder = "./save" + language_path
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        audio_clips_path = common_voice_path + language_path + "/clips"
        
        # Acquire list of audio clips, cap at maximum
        audioClips = os.listdir(audio_clips_path)
        numAudioClips = len(audioClips)
        if numAudioClips > maxSamples:
            audioClips = audioClips[0:maxSamples]
            print("Creating CSV files for %s: %d samples (clipped from %d)" %(language_path, len(audioClips), numAudioClips))
        else:
            print("Creating CSV files for %s: %d samples" %(language_path, len(audioClips)))

        # Shuffle and divide into training, validation, and testing sets
        random.shuffle(audioClips)
        num_train_samples = int(0.8 * len(audioClips))
        num_dev_samples = int(0.1 * len(audioClips))
        num_test_samples = int(0.1 * len(audioClips))
        train_idx_start = 0
        train_idx_end = train_idx_start + num_train_samples
        dev_idx_start = train_idx_end + 1
        dev_idx_end = dev_idx_start + num_dev_samples
        test_idx_start = dev_idx_end + 1
        test_idx_end = test_idx_start + num_test_samples
        # Clip the ending testing index
        if test_idx_end > len(audioClips):
            test_idx_end = len(audioClips)

        # Create train.csv, dev.csv, test.csv
        createCsvFile(audioClips, train_idx_start, train_idx_end, save_folder + "/train.csv")
        createCsvFile(audioClips, dev_idx_start, dev_idx_end, save_folder + "/dev.csv")
        createCsvFile(audioClips, test_idx_start, test_idx_end, save_folder + "/test.csv")
        
    return

# Takes contents of CSV file to convert audio from mp3 to wav format
def prepData(common_voice_path, csv_file_name, data_path, data_type):
    for language_path in LANGUAGE_PATHS:
        print("Setting up %s data for %s" %(data_type, language_path))
        language = language_path.split("/")[2]
        csv_file_path = "./save" + language_path + "/" + csv_file_name

        # Count number of rows/mp3 files
        numRows = 0
        with open(csv_file_path) as csv_file:
            rows = csv.reader(csv_file, delimiter=',')
            numRows = sum(1 for row in rows)

        # Convert the .mp3 file into .wav and store it only if the .wav file is not already present
        with open(csv_file_path) as csv_file:
            rows = csv.reader(csv_file, delimiter=',')
            dataCnt = 0
            for row in rows:
                mp3_file = row[0]
                wav_file = mp3_file.split(".")[0] + ".wav"
                lang_path = data_path + "/" + language
                if not os.path.exists(lang_path):
                    os.mkdir(lang_path)
                dest_path = lang_path + "/" + wav_file
                audio_clips_path = common_voice_path + language_path + "/clips"

                if not os.path.exists(dest_path):
                    sound = AudioSegment.from_mp3(audio_clips_path + "/" + mp3_file)
                    sound.export(wav_file, format="wav")
                    shutil.move("./" + wav_file, dest_path)
                
                # Check if max is reached, print status
                dataCnt = dataCnt + 1
                if 0 == dataCnt % 1000 or 0 == dataCnt or numRows == dataCnt:
                    print("%d/%d processed" %(dataCnt, numRows))
    return

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', type=str, required=True, help="Path to CommonVoice dataset")
    parser.add_argument('-max_samples', type=int, default=1000, help="Max number of samples used for training, validation, and testing per language")
    parser.add_argument('--createCsv', action="store_true", default=False, help="Creates CSV files for training, validation, and testing")
    parser.add_argument('--train', action="store_true", default=False, help="Prepare training data")
    parser.add_argument('--dev', action="store_true", default=False, help="Prepare validation/dev data")
    parser.add_argument('--test', action="store_true", default=False, help="Prepare testing data")
    args = parser.parse_args()

    commonVoicePath = args.path
    maxSamples = args.max_samples
    prepTrainData = args.train
    prepDevData = args.dev
    prepTestData = args.test
    createCsv = args.createCsv
    
    # Data paths
    TRAIN_PATH = commonVoicePath + "/commonVoice/train"
    TEST_PATH = commonVoicePath + "/commonVoice/test"
    DEV_PATH = commonVoicePath + "/commonVoice/dev"

    # Prepare the csv files for the Common Voice dataset
    if createCsv:
        createAllCsvFiles(commonVoicePath, maxSamples)
        
        if os.path.exists(TRAIN_PATH):
            print("Cleaning up %s folder" %TRAIN_PATH)
            shutil.rmtree(TRAIN_PATH)
        os.makedirs(TRAIN_PATH)
        
        if os.path.exists(DEV_PATH):
            print("Cleaning up %s folder" %DEV_PATH)
            shutil.rmtree(DEV_PATH)
        os.makedirs(DEV_PATH)

        if os.path.exists(TEST_PATH):
            print("Cleaning up %s folder" %TEST_PATH)
            shutil.rmtree(TEST_PATH)
        os.makedirs(TEST_PATH)

    # Generate the training, validation, and testing datasets
    if prepTrainData:
        print("\n\nPREPARING TRAINING DATA")
        prepData(commonVoicePath, "train.csv", TRAIN_PATH, "training")
    if prepDevData:
        print("\n\nPREPARING VALIDATION/DEV DATA")
        prepData(commonVoicePath, "dev.csv", DEV_PATH, "validation")
    if prepTestData:
        print("\n\nPREPARING TESTING DATA")
        prepData(commonVoicePath, "test.csv", TEST_PATH, "testing")

    return


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
