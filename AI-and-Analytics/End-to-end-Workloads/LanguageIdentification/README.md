# `Language Identification` Sample

This `Language Identification` sample demonstrates how to train a model to perform language identification using the Hugging Face SpeechBrain speech toolkit.

Languages are selected from the CommonVoice dataset for training, validation, and testing.

| Area                  | Description
|:---                   |:---
| What you will learn   | How to use training and inference with SpeechBrain, Intel® Extension for PyTorch* (IPEX) inference, Intel® Neural Compressor (INC) quantization
| Time to complete      | 60 minutes

## Purpose

Spoken audio comes in different languages and this sample uses a model to identify what that language is. The user will use an Intel® AI Analytics Toolkit container environment to train a model and perform inference leveraging Intel-optimized libraries for PyTorch*. There is also an option to quantize the trained model with Intel® Neural Compressor (INC) to speed up inference.

## Prerequisites

| Optimized for        | Description
|:---                  |:---
| OS                   | Ubuntu* 22.04 or newer
| Hardware             | Intel® Xeon® and Core® processor families
| Software             | Intel® AI Tools <br> Hugging Face SpeechBrain

## Key Implementation Details

The [CommonVoice](https://commonvoice.mozilla.org/) dataset is used to train an Emphasized Channel Attention, Propagation and Aggregation Time Delay Neural Network (ECAPA-TDNN). This is implemented in the [Hugging Face SpeechBrain](https://huggingface.co/SpeechBrain) library. Additionally, a small Convolutional Recurrent Deep Neural Network (CRDNN) pretrained on the LibriParty dataset is used to process audio samples and output the segments where speech activity is detected.

The model is then trained from scratch using the Hugging Face SpeechBrain library. This model is then used for inference on the testing dataset or a user-specified dataset. There is an option to utilize SpeechBrain's Voice Activity Detection (VAD) where only the speech segments from the audio files are extracted and combined before samples are randomly selected as input into the model. To improve performance, the user may quantize the trained model to INT8 using Intel® Neural Compressor (INC) to decrease latency.

The sample contains three discreet phases:
- [Prepare the Environment](#prepare-the-environment)
- [Train the Model with Languages](#train-the-model-with-languages)
- [Run Inference for Language Identification](#run-inference-for-language-identification)

For both training and inference, you can run the sample and scripts in Jupyter Notebook or you can choose to run the sample locally and directly. The relevant sections provide instructions for both options.


## Prepare the Environment

### Create and Set Up Environment

1. Create your conda environment by following the instructions on the Intel [AI Tools Selector](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-tools-selector.html). You can follow these settings:

* Tool: AI Tools
* Preset or customize: Customize
* Distribution Type: conda* or pip
* Python Versions: Python* 3.9 or 3.10
* PyTorch* Framework Optimizations: Intel® Extension for PyTorch* (CPU)
* Intel®-Optimized Tools & Libraries: Intel® Neural Compressor

>**Note**: Be sure to activate your environment before installing the packages. If using pip, install using `python -m pip` instead of just `pip`.

2. Create your dataset folder and set the environment variable `COMMON_VOICE_PATH`. This needs to match with where you downloaded your dataset.
```bash
mkdir -p /data/commonVoice
export COMMON_VOICE_PATH=/data/commonVoice
```

3. Install packages needed for MP3 to WAV conversion
```bash
sudo apt-get update && apt-get install -y ffmpeg libgl1
```

4. Navigate to your working directory, clone the `oneapi-src` repository, and navigate to this code sample.
```bash
git clone https://github.com/oneapi-src/oneAPI-samples.git 
cd oneAPI-samples/AI-and-Analytics/End-to-end-Workloads/LanguageIdentification 
```

5. Run the bash script to install additional necessary libraries, including SpeechBrain.
```bash
source initialize.sh
```

### Download the CommonVoice Dataset

>**Note**: You can skip downloading the dataset if you already have a pretrained model and only want to run inference on custom data samples that you provide.

First, change to the `Dataset` directory.
```
cd ./Dataset
```

The `get_dataset.py` script downloads the Common Voice dataset by doing the following:

- Gets the train set of the [Common Voice dataset from Huggingface](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) for Japanese and Swedish
- Downloads each mp3 and moves them to the `output_dir` folder

1. If you want to add additional languages, then modify the `language_to_code` dictionary in the file to reflect the languages to be included in the model.

3. Run the script with options.
   ```bash
   python get_dataset.py --output_dir ${COMMON_VOICE_PATH}
   ```
   | Parameters      | Description
   |:---             |:---
   | `--output_dir`  | Base output directory for saving the files. Default is /data/commonVoice

Once the dataset is downloaded, navigate back to the parent directory
```
cd ..
```

## Train the Model with Languages

This section explains how to train a model for language identification using the CommonVoice dataset, so it includes steps on how to preprocess the data, train the model, and prepare the output files for inference.

First, change to the `Training` directory.
```
cd ./Training
```

### Option 1: Run in Jupyter Notebook

1. Launch Jupyter Notebook.
   ```
   jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root
   ```
2. Follow the instructions to open the URL with the token in your browser.
3. Locate and select the Training Notebook.
   ```
   lang_id_training.ipynb
   ```
4. Follow the instructions in the Notebook.


### Option 2: Run in a Console

If you cannot or do not want to use Jupyter Notebook, use these procedures to run the sample and scripts locally.

#### Process the Data

1. Acquire copies of the training scripts. (The command retrieves copies of the required VoxLingua107 training scripts from SpeechBrain.)
   ```
   cp ../speechbrain/recipes/VoxLingua107/lang_id/create_wds_shards.py create_wds_shards.py
   cp ../speechbrain/recipes/VoxLingua107/lang_id/train.py train.py
   cp ../speechbrain/recipes/VoxLingua107/lang_id/hparams/train_ecapa.yaml train_ecapa.yaml
   ```

2. From the `Training` directory, apply patches to modify these files to work with the CommonVoice dataset.
   ```bash
   patch < create_wds_shards.patch
   patch < train_ecapa.patch
   ```

#### Create Datasets for Training, Validation, and Testing

The `prepareAllCommonVoice.py` script performs the following data preprocessing steps:

- Generates .csv files for training, validation, and testing and puts them into the **/save** folder.
- Opens each .csv file to get the path to the **.mp3** file, and converts the file to **.wav** format.

1. If you want to add additional languages, then modify the `LANGUAGE_PATHS` list in the file to reflect the languages to be included in the model.

2. Run the script with options. The samples will be divided as follows: 80% training, 10% validation, 10% testing.
   ```bash
   python prepareAllCommonVoice.py -path $COMMON_VOICE_PATH -max_samples 2000 --createCsv --train --dev --test
   ```
   | Parameters      | Description
   |:---             |:---
   | `-max_samples`  | The maximum number of samples used for training, validation, and testing. If no value is specified, the sample uses the default of **1000**. If `max_samples` is set to a value greater than the number of available samples for a language, the script will use the upper limit instead. This example uses **2000**.
   | `--createCsv`   | **Use the option ONCE only.** Use the `--createCsv` option only if you want to create new training/dev/test sets. Remove the option for all subsequent runs; otherwise, all converted .wav files will be deleted and you must restart preprocessing.

>**Note**: You should create multiple versions of `prepareAllCommonVoice.py` and spawn multiple terminals to execute on different languages because the process takes a long time to complete. Introducing threading can also speed up the process.

#### Create Shards for Training and Validation

1. If the `${COMMON_VOICE_PATH}/processed_data/commonVoice_shards` folder exists, delete the folder and the contents before proceeding.
2. Enter the following commands.
   ```bash
   python create_wds_shards.py ${COMMON_VOICE_PATH}/processed_data/train ${COMMON_VOICE_PATH}/processed_data/commonVoice_shards/train
   python create_wds_shards.py ${COMMON_VOICE_PATH}/processed_data/dev ${COMMON_VOICE_PATH}/processed_data/commonVoice_shards/dev
   ```
3. Note the shard with the largest number as `LARGEST_SHARD_NUMBER` in the output above or by navigating to `${COMMON_VOICE_PATH}/processed_data/commonVoice_shards/train`.
4. Open the `train_ecapa.yaml` file and modify the `train_shards` variable to make the range reflect: `000000..LARGEST_SHARD_NUMBER`.
5. Repeat Steps 3 and 4 for `${COMMON_VOICE_PATH}/processed_data/commonVoice_shards/dev`.

#### Run the Training Script

The YAML file `train_ecapa.yaml` with the training configurations is passed as an argument to the `train.py` script to train the model.

1. If necessary, edit the `train_ecapa.yaml` file to meet your needs.

   | Parameters          | Description
   |:---                 |:---
   | `seed`              | The seed value, which should be set to a different value for subsequent runs. Defaults to 1987.
   | `out_n_neurons`     | Must be equal to the number of languages of interest.
   | `number_of_epochs`  | Default is **10**. Adjust as needed.
   | `batch_size`        | In the trainloader_options, decrease this value if your CPU or GPU runs out of memory while running the training script. If you see a "Killed" error message, then the training script has run out of memory. 

2. Run the script to train the model.
   ```
   python train.py train_ecapa.yaml --device "cpu"
   ```

#### Move Model to Inference Folder

After training, the output should be inside the `results/epaca/1987` folder. By default the `seed` is set to 1987 in `train_ecapa.yaml`. You can change the value as needed.

1. Copy all files from `results/epaca/1987` into a new folder called `lang_id_commonvoice_model` in the **Inference** folder.
   ```bash
   cp -R results/epaca/1987 ../Inference/lang_id_commonvoice_model
   ```
   The name of the folder MUST match with the pretrained_path variable defined in `train_ecapa.yaml`. By default, it is `lang_id_commonvoice_model`.

2. Change directory to `/Inference/lang_id_commonvoice_model/save`.
   ```bash
   cd ../Inference/lang_id_commonvoice_model/save
   ```

3. Copy the `label_encoder.txt` file up one level.
   ```bash
   cp label_encoder.txt ../.
   ```

4. Change to the latest `CKPT` folder, and copy the classifier.ckpt and embedding_model.ckpt files into the `/Inference/lang_id_commonvoice_model/` folder which is two directories up. By default, the command below will navigate into the single CKPT folder that is present, but you can change it to the specific folder name. 
   ```bash
   # Navigate into the CKPT folder
   cd CKPT*

   cp classifier.ckpt ../../.
   cp embedding_model.ckpt ../../
   cd ../../../..
   ```

   You may need to modify the permissions of these files to be executable i.e. `sudo chmod 755` before you run the inference scripts to consume them.

>**Note**: If `train.py` is rerun with the same seed, it will resume from the epoch number it last run. For a clean rerun, delete the `results` folder or change the seed.

You can now load the model for inference. In the `Inference` folder, the `inference_commonVoice.py` script uses the trained model on the testing dataset, whereas `inference_custom.py` uses the trained model on a user-specified dataset and can utilize Voice Activity Detection. 

>**Note**: If the folder name containing the model is changed from `lang_id_commonvoice_model`, you will need to modify the `pretrained_path` in `train_ecapa.yaml`, and the `source_model_path` variable in both the `inference_commonVoice.py` and `inference_custom.py` files in the `speechbrain_inference` class. 


## Run Inference for Language Identification

>**Stop**: If you have not already done so, you must run the scripts in the `Training` folder to generate the trained model before proceeding.

To run inference, you must have already run all of the training scripts, generated the trained model, and moved files to the appropriate locations. You must place the model output in a folder name matching the name specified as the `pretrained_path` variable defined in `train_ecapa.yaml`.

>**Note**: If you plan to run inference on **custom data**, you will need to create a folder for the **.wav** files to be used for prediction. For example, `data_custom`. Move the **.wav** files to your custom folder. (For quick results, you may select a few audio files from each language downloaded from CommonVoice.)

### Configure the Inference Environment

1. Change to the `Inference` directory.
   ```
   cd /Inference
   ```

### Option 1: Run in Jupyter Notebook

1. Launch Jupyter Notebook.
   ```
   jupyter notebook --ip 0.0.0.0 --port 8889 --allow-root
   ```
2. Follow the instructions to open the URL with the token in your browser.
3. Locate and select the inference Notebook.
   ```
   lang_id_inference.ipynb
   ```
4. Follow the instructions in the Notebook.

### Option 2: Run in a Console

If you cannot or do not want to use Jupyter Notebook, use these procedures to run the sample and scripts locally.

### Inference Scripts and Options

You can run inference on user-supplied data or on the CommonVoice dataset. The `Inference` folder contains scripts for each scenario:

| Script                      | Description
|:---                         |:---
| `inference_custom.py`       | Run this script to run inference on user-specified data
| `inference_commonVoice.py`  | Run this script to run inference on the testing dataset from the CommonVoice dataset.

Both scripts support input options; however, some options can be use on `inference_custom.py` only.

| Input Option               | Description
|:---                        |:---
| `-p`                       | Specify the datapath.
| `-d`                       | Specify the duration of wave sample. Default value is **3**.
| `-s`                       | Specify size of sample waves, default is **100**.
| `--vad`                    | (`inference_custom.py` only) Enable VAD model to detect active speech. The VAD option will identify speech segments in the audio file and construct a new **.wav** file containing only the speech segments. This improves the quality of speech data used as input into the language identification model.
| `--ipex`                   | Run inference with optimizations from Intel® Extension for PyTorch (IPEX). This option will apply optimizations to the pretrained model. Using this option should result in performance improvements related to latency.
| `--bf16`                   | Run inference with auto-mixed precision featuring Bfloat16.
| `--int8_model`             | Run inference with the INT8 model generated from Intel® Neural Compressor (INC)
| `--ground_truth_compare`   | (`inference_custom.py` only) Enable comparison of prediction labels to ground truth values.
| `--verbose`                | Print additional debug information, like latency.


### Run Inference 

#### On the CommonVoice Dataset

1. Run the inference_commonvoice.py script.
   ```bash
   python inference_commonVoice.py -p ${COMMON_VOICE_PATH}/processed_data/test
   ```
   The script should create a `test_data_accuracy.csv` file that summarizes the results.

#### On Custom Data

To run inference on custom data, you must specify a folder with **.wav** files and pass the path in as an argument. You can do so by creating a folder named `data_custom` and then copy 1 or 2 **.wav** files from your test dataset into it. **.mp3** files will NOT work. 

Run the inference_ script.
```bash
python inference_custom.py -p <path_to_folder>
```

The following examples describe how to use the scripts to produce specific outcomes.

**Default: Random Selections**

1. To randomly select audio clips from audio files for prediction, enter commands similar to the following:
   ```bash
   python inference_custom.py -p data_custom -d 3 -s 50
   ```
   This picks 50 3-second samples from each **.wav** file in the `data_custom` folder. The `output_summary.csv` file summarizes the results.

2. To randomly select audio clips from audio files after applying **Voice Activity Detection (VAD)**, use the `--vad` option:
   ```bash
   python inference_custom.py -p data_custom -d 3 -s 50 --vad
   ```
   Again, the `output_summary.csv` file summarizes the results. 

   >**Note**: The audio input into the VAD model must be sampled at **16kHz**. The code performs this conversion by default.

**Optimization with Intel® Extension for PyTorch (IPEX)**

1. To optimize user-defined data, enter commands similar to the following:
   ```bash
   python inference_custom.py -p data_custom -d 3 -s 50 --vad --ipex --verbose
   ```
   This will apply `ipex.optimize` to the model(s) and TorchScript. You can also add the `--bf16` option along with `--ipex` to run in the BF16 data type, supported on 4th Gen Intel® Xeon® Scalable processors and newer.
   
   >**Note**: The `--verbose` option is required to view the latency measurements.

**Quantization with Intel® Neural Compressor (INC)**

1. To improve inference latency, you can use the Intel® Neural Compressor (INC) to quantize the trained model from FP32 to INT8 by running `quantize_model.py`.
   ```bash
   python quantize_model.py -p ./lang_id_commonvoice_model -datapath $COMMON_VOICE_PATH/processed_data/dev
   ```
   Use the `-datapath` argument to specify a custom evaluation dataset. By default, the datapath is set to the `$COMMON_VOICE_PATH/processed_data/dev` folder that was generated from the data preprocessing scripts in the `Training` folder.

   After quantization, the model will be stored in `lang_id_commonvoice_model_INT8` and `neural_compressor.utils.pytorch.load` will have to be used to load the quantized model for inference. If `self.language_id` is the original model and `data_path` is the path to the audio file:
   ```
   from neural_compressor.utils.pytorch import load
   model_int8 = load("./lang_id_commonvoice_model_INT8", self.language_id)
   signal = self.language_id.load_audio(data_path)
   prediction = self.model_int8(signal)
   ```

   The code above is integrated into `inference_custom.py`. You can now run inference on your data using this INT8 model:
   ```bash
   python inference_custom.py -p data_custom -d 3 -s 50 --vad --int8_model --verbose
   ```

   >**Note**: The `--verbose` option is required to view the latency measurements.

**(Optional) Comparing Predictions with Ground Truth**

You can choose to modify `audio_ground_truth_labels.csv` to include the name of the audio file and expected audio label (like, `en` for English), then run `inference_custom.py` with the `--ground_truth_compare` option. By default, this is disabled.  

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
