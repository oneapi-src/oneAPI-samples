# `Language Identification` Sample

This `Language Identification` sample demonstrates how to train a model to perform language identification using the Hugging Face SpeechBrain speech toolkit.

Languages are selected from the CommonVoice dataset for training, validation, and testing.

| Area                  | Description
|:---                   |:---
| What you will learn   | How to use training and inference with SpeechBrain, Intel® Extension for PyTorch* inference, Intel® Neural Compressor quantization, and a oneapi-aikit container
| Time to complete      | 60 minutes

## Purpose

Spoken audio comes in different languages and this sample uses a model to identify what that language is. The user will use an Intel® AI Analytics Toolkit container environment to train a model and perform inference leveraging Intel-optimized libraries for PyTorch*. There is also an option to quantize the trained model with Neural Compressor to speed up inference.

## Prerequisites

| Optimized for        | Description
|:---                  |:---
| OS                   | Ubuntu* 18.04 or newer
| Hardware             | Intel® Xeon® processor family
| Software             | Intel® OneAPI AI Analytics Toolkit <br> Hugging Face SpeechBrain

## Key Implementation Details

The [CommonVoice](https://commonvoice.mozilla.org/) dataset is used to train an Emphasized Channel Attention, Propagation and Aggregation Time Delay Neural Network (ECAPA-TDNN). This is implemented in the [Hugging Face SpeechBrain](https://huggingface.co/SpeechBrain) library. Additionally, a small Convolutional Recurrent Deep Neural Network (CRDNN) pretrained on the LibriParty dataset is used to process audio samples and output the segments where speech activity is detected.

After you have downloaded the CommonVoice dataset, the data must be preprocessed by converting the MP3 files into WAV format and separated into training, validation, and testing sets.

The model is then trained from scratch using the Hugging Face SpeechBrain library. This model is then used for inference on the testing dataset or a user-specified dataset. There is an option to utilize SpeechBrain's Voice Activity Detection (VAD) where only the speech segments from the audio files are extracted and combined before samples are randomly selected as input into the model. To improve performance, the user may quantize the trained model to INT8 using Neural Compressor to decrease latency.

The sample contains three discreet phases:
- [Prepare the Environment](#prepare-the-environment)
- [Train the Model with Languages](#train-the-model-with-languages)
- [Run Inference for Language Identification](#run-inference-for-language-identification)

For both training and inference, you can run the sample and scripts in Jupyter Notebook or you can choose to run the sample locally and directly. The relevant sections provide instructions for both options.


## Prepare the Environment

### Downloading the CommonVoice Dataset

>**Note**: You can skip downloading the dataset if you already have a pretrained model and only want to run inference on custom data samples that you provide.

Download the CommonVoice dataset for languages of interest from [https://commonvoice.mozilla.org/en/datasets](https://commonvoice.mozilla.org/en/datasets). 

For this sample, you will need to download the following languages: **Japanese** and **Swedish**. Follow Steps 1-6 below or you can execute the code.  

1. On the CommonVoice website, select the Version and Language.
2. Enter your email.
3. Check the boxes, and right-click on the download button to copy the link address.
4. Paste this link into a text editor and copy the first part of the URL up to ".tar.gz".
5. Use **GNU wget** on the URL to download the data to `/data/commonVoice`.

   Alternatively, you can use a directory on your local drive (due to the large amount of data). If you opt to do so, you must change the `COMMON_VOICE_PATH` environment in `launch_docker.sh` before running the script.

6. Extract the compressed folder, and rename the folder with the language (for example, English).

   The file structure **must match** the `LANGUAGE_PATHS` defined in `prepareAllCommonVoice.py` in the `Training` folder for the script to run properly.

These commands illustrate Steps 1-6. Notice that it downloads Japanese and Swedish from CommonVoice version 11.0.  
```
# Create the commonVoice directory under 'data'
sudo chmod 777 -R /data
cd /data
mkdir commonVoice
cd commonVoice

# Download the CommonVoice data
wget \
https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-11.0-2022-09-21/cv-corpus-11.0-2022-09-21-ja.tar.gz \
https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-11.0-2022-09-21/cv-corpus-11.0-2022-09-21-sv-SE.tar.gz

# Extract and organize the CommonVoice data into respective folders by language 
tar -xf cv-corpus-11.0-2022-09-21-ja.tar.gz
mv cv-corpus-11.0-2022-09-21 japanese
tar -xf cv-corpus-11.0-2022-09-21-sv-SE.tar.gz
mv cv-corpus-11.0-2022-09-21 swedish
```

### Configuring the Container

1. Pull the `oneapi-aikit` docker image.
2. Set up the Docker environment.
   ```
   docker pull intel/oneapi-aikit
   ./launch_docker.sh
   ```
   >**Note**: By default, the `Inference` and `Training` directories will be mounted and the environment variable `COMMON_VOICE_PATH` will be set to `/data/commonVoice` and mounted to `/data`. `COMMON_VOICE_PATH` is the location of where the CommonVoice dataset is downloaded.



## Train the Model with Languages

This section explains how to train a model for language identification using the CommonVoice dataset, so it includes steps on how to preprocess the data, train the model, and prepare the output files for inference.

### Configure the Training Environment

1. Change to the `Training` directory.
   ```
   cd /Training
   ```
2. Source the bash script to install the necessary components.
   ```
   source initialize.sh
   ```
   This installs PyTorch*, the Intel® Extension for PyTorch*, and other components.

### Run in Jupyter Notebook

1. Install Jupyter Notebook.
   ```
   conda install jupyter nb_conda_kernels
   ```
2. Launch Jupyter Notebook.
   ```
   jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root
   ```
3. Follow the instructions to open the URL with the token in your browser.
4. Locate and select the Training Notebook.
   ```
   lang_id_training.ipynb
   ```
5. Follow the instructions in the Notebook.


### Run in a Console

If you cannot or do not want to use Jupyter Notebook, use these procedures to run the sample and scripts locally.

#### Process the Data

1. Acquire copies of the training scripts. (The command retrieves copies of the required VoxLingua107 training scripts from SpeechBrain.)
   ```
   cp speechbrain/recipes/VoxLingua107/lang_id/create_wds_shards.py create_wds_shards.py
   cp speechbrain/recipes/VoxLingua107/lang_id/train.py train.py
   cp speechbrain/recipes/VoxLingua107/lang_id/hparams/train_ecapa.yaml train_ecapa.yaml
   ```

2. From the `Training` directory, apply patches to modify these files to work with the CommonVoice dataset.
   ```
   patch < create_wds_shards.patch
   patch < train_ecapa.patch
   ```

#### Create Datasets for Training, Validation, and Testing

The `prepareAllCommonVoice.py` script performs the following data preprocessing steps:

- Generates .csv files for training, validation, and testing and puts them into the **/save** folder.
- Opens each .csv file to get the path to the **.mp3** file, and converts the file to **.wav** format.

1. If you want to add additional languages, then modify the `LANGUAGE_PATHS` list in the file to reflect the languages to be included in the model.

2. Run the script with options. The samples will be divided as follows: 80% training, 10% validation, 10% testing.
   ```
   python prepareAllCommonVoice.py -path /data -max_samples 2000 --createCsv --train --dev --test
   ```
   | Parameters      | Description
   |:---             |:---
   | `-max_samples`  | The maximum number of samples used for training, validation, and testing. If no value is specified, the sample uses the default of **1000**. If `max_samples` is set to a value greater than the number of available samples for a language, the script will use the upper limit instead. This example uses **2000**.
   | `--createCsv`   | **Use the option ONCE only.** Use the `--createCsv` option only if you want to create new training/dev/test sets. Remove the option for all subsequent runs; otherwise, all converted .wav files will be deleted and you must restart preprocessing.

>**Note**: You should create multiple versions of `prepareAllCommonVoice.py` and spawn multiple terminals to execute on different languages because the process takes a long time to complete. Introducing threading can also speed up the process.

#### Create Shards for Training and Validation

1. If the `/data/commonVoice_shards` folder exists, delete the folder and the contents before proceeding.
2. Enter the following commands.
   ```
   python create_wds_shards.py /data/commonVoice/train/ /data/commonVoice_shards/train
   python create_wds_shards.py /data/commonVoice/dev/ /data/commonVoice_shards/dev
   ```
3. Note the shard with the largest number as `LARGEST_SHARD_NUMBER` in the output above or by navigating to `/data/commonVoice_shards/train`.
4. Open the `train_ecapa.yaml` file and modify the `train_shards` variable to make the range reflect: `000000..LARGEST_SHARD_NUMBER`.
5. Repeat the process for `/data/commonVoice_shards/dev`.

#### Run the Training Script

The YAML file `train_ecapa.yaml` with the training configurations should already be patched from the Prerequisite section.

1. If necessary, edit the `train_ecapa.yaml` file to meet your needs.

   | Parameters          | Description
   |:---                 |:---
   | `out_n_neurons`     | Must be equal to the number of languages of interest.
   | `number_of_epochs`  | Default is **10**. Adjust as needed.
   | `batch_size`        | In the trainloader_options, decrease this value if your CPU or GPU runs out of memory while running the training script.

2. Run the script to train the model.
   ```
   python train.py train_ecapa.yaml --device "cpu"
   ```

#### Move Model to Inference Folder

After training, the output should be inside `results/epaca/SEED_VALUE` folder. By default SEED_VALUE is set to 1987 in the YAML file. You can change the value as needed.

1. Copy all files with *cp -R* from `results/epaca/SEED_VALUE` into a new folder called `lang_id_commonvoice_model` in the **Inference** folder.

   The name of the folder MUST match with the pretrained_path variable defined in the YAML file. By default, it is `lang_id_commonvoice_model`.

2. Change directory to `/Inference/lang_id_commonvoice_model/save`.
3. Copy the `label_encoder.txt` file up one level.
4. Change to the latest `CKPT` folder, and copy the classifier.ckpt and embedding_model.ckpt files into the `/Inference/lang_id_commonvoice_model/` folder.

   You may need to modify the permissions of these files to be executable before you run the inference scripts to consume them.

>**Note**: If `train.py` is rerun with the same seed, it will resume from the epoch number it last run. For a clean rerun, delete the `results` folder or change the seed.

You can now load the model for inference. In the `Inference` folder, the `inference_commonVoice.py` script uses the trained model on the testing dataset, whereas `inference_custom.py` uses the trained model on a user-specified dataset and can utilize Voice Activity Detection. 

>**Note**: If the folder name containing the model is changed from `lang_id_commonvoice_model`, you will need to modify the `source_model_path` variable in `inference_commonVoice.py` and `inference_custom.py` files in the `speechbrain_inference` class.


## Run Inference for Language Identification

>**Stop**: If you have not already done so, you must run the scripts in the `Training` folder to generate the trained model before proceeding.

To run inference, you must have already run all of the training scripts, generated the trained model, and moved files to the appropriate locations. You must place the model output in a folder name matching the name specified as the `pretrained_path` variable defined in the YAML file.

>**Note**: If you plan to run inference on **custom data**, you will need to create a folder for the **.wav** files to be used for prediction. For example, `data_custom`. Move the **.wav** files to your custom folder. (For quick results, you may select a few audio files from each language downloaded from CommonVoice.)

### Configure the Inference Environment

1. Change to the `Inference` directory.
   ```
   cd /Inference
   ```
2. Source the bash script to install or update the necessary components.
   ```
   source initialize.sh
   ```
3. Patch the Intel® Extension for PyTorch* to use SpeechBrain models. (This patch is required for PyTorch* TorchScript to work because the output of the model must contain only tensors.)
   ```
   patch ./speechbrain/speechbrain/pretrained/interfaces.py < interfaces.patch
   ```

### Run in Jupyter Notebook

1. If you have not already done so, install Jupyter Notebook.
   ```
   conda install jupyter nb_conda_kernels
   ```
2. Launch Jupyter Notebook.
   ```
   jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root
   ```
3. Follow the instructions to open the URL with the token in your browser.
4. Locate and select the inference Notebook.
   ```
   lang_id_inference.ipynb
   ```
5. Follow the instructions in the Notebook.

### Run in a Console

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
| `--ipex`                   | Run inference with optimizations from Intel® Extension for PyTorch*. This option will apply optimizations to the pretrained model. Using this option should result in performance improvements related to latency.
| `--bf16`                   | Run inference with auto-mixed precision featuring Bfloat16.
| `--int8_model`             | Run inference with the INT8 model generated from Intel® Neural Compressor
| `--ground_truth_compare`   | (`inference_custom.py` only) Enable comparison of prediction labels to ground truth values.
| `--verbose`                | Print additional debug information, like latency.


### Run Inference 

#### On the CommonVoice Dataset

1. Run the inference_commonvoice.py script.
   ```
   python inference_commonVoice.py -p /data/commonVoice/test
   ```
   The script should create a `test_data_accuracy.csv` file that summarizes the results.

#### On Custom Data

1. Modify the `audio_ground_truth_labels.csv` file to include the name of the audio file and expected audio label (like, `en` for English).

   By default, this is disabled. If required, use the `--ground_truth_compare` input option. To run inference on custom data, you must specify a folder with **.wav** files and pass the path in as an argument.

2. Run the inference_ script.
   ```
   python inference_custom.py -p <data path>
   ```

The following examples describe how to use the scripts to produce specific outcomes.

**Default: Random Selections**

1. To randomly select audio clips from audio files for prediction, enter commands similar to the following:
   ```
   python inference_custom.py -p data_custom -d 3 -s 50
   ```
   This picks 50 3-second samples from each **.wav** file in the `data_custom` folder. The `output_summary.csv` file summarizes the results.

2. To randomly select audio clips from audio files after applying **Voice Activity Detection (VAD)**, use the `--vad` option:
   ```
   python inference_custom.py -p data_custom -d 3 -s 50 --vad
   ```
   Again, the `output_summary.csv` file summarizes the results. 

   >**Note**: The audio input into the VAD model must be sampled at **16kHz**. The code performs this conversion by default.

**Optimization with Intel® Extension for PyTorch***

1. To optimize user-defined data, enter commands similar to the following:
   ```
   python inference_custom.py -p data_custom -d 3 -s 50 --vad --ipex --verbose
   ```
   >**Note**: The `--verbose` option is required to view the latency measurements.

**Quantization with Neural Compressor**

1. To improve inference latency, you can use the Neural Compressor to quantize the trained model from FP32 to INT8 by running `quantize_model.py`.
   ```
   python quantize_model.py -p ./lang_id_commonvoice_model -datapath $COMMON_VOICE_PATH/dev
   ```
   Use the `-datapath` argument to specify a custom evaluation dataset. By default, the datapath is set to the `/data/commonVoice/dev` folder that was generated from the data preprocessing scripts in the `Training` folder.

   After quantization, the model will be stored in `lang_id_commonvoice_model_INT8` and `neural_compressor.utils.pytorch.load` will have to be used to load the quantized model for inference. If `self.language_id` is the original model and `data_path` is the path to the audio file:
   ```
   from neural_compressor.utils.pytorch import load
   model_int8 = load("./lang_id_commonvoice_model_INT8", self.language_id)
   signal = self.language_id.load_audio(data_path)
   prediction = self.model_int8(signal)
   ```

### Troubleshooting

If the model appears to be giving the same output regardless of input, try running `clean.sh` to remove the `RIR_NOISES` and `speechbrain` folders. Redownload that data after cleaning by running `initialize.sh` and either `inference_commonVoice.py` or `inference_custom.py`.

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).