# Language Identification Sample  

This sample code trains a model to perform language identification using the Hugging Face SpeechBrain speech toolkit. Languages are selected from the CommonVoice dataset for training, validation, and testing.  

| Optimized for                     | Description  
| :---                              | :---  
| OS                                | 64-bit Linux: Ubuntu 18.04 or higher  
| Hardware                          | Intel® Xeon® Scalable processor family  
| Software                          | Intel® OneAPI AI Analytics Toolkit, Hugging Face Speechbrain  
| What you will learn               | oneapi-aikit container, training and inference with Speechbrain, IPEX inference, INC quantization  
| Time to complete                  | 60 minutes  

## Purpose  
Spoken audio comes in different languages and this sample uses a model to identify what that language is. The user will use an Intel® AI Analytics Toolkit container 
environment to train a model and perform inference leveraging Intel-optimized libraries for PyTorch. There is also an option to quantize the trained model with 
Neural Compressor to speed up inference.  

#### Model and Dataset  
The CommonVoice dataset is used to train an Emphasized Channel Attention, Propagation and Aggregation Time Delay Neural Network (ECAPA-TDNN).  This is implemented 
in Hugging Face's Speechbrain library. In addition, a small Convolutional Recurrent Deep Neural Network (CRDNN) pretrained on the LibriParty dataset is used to process audio samples and output the segments where speech activity is detected.  

## Key Implementation Details  
After the CommonVoice dataset is downloaded, the data must be preprocessed by converting the MP3 files into WAV format and separated into training, validation, and 
testing sets. The model is then trained from scratch using Hugging Face's Speechbrain library. This model is then used for inference on the testing dataset or a 
user-specified dataset. There is an option to utilize Speechbrain's Voice Activity Detection (VAD) where only the speech segments from the audio files are extracted 
and combined before samples are randomly selected as input into the model. To improve performance, the user may quantize the trained model to INT8 using Neural 
Compressor to decrease latency.  

## Running the Sample  
### Downloading the CommonVoice Dataset  
** Note ** You may skip the dataset download if you already have a pretrained model and only want to run inference on custom data samples that you provide.  

Download the CommonVoice dataset for languages of interest: https://commonvoice.mozilla.org/en/datasets. For this sample, download the following languages:  

| Languages                    | Japanese, Swedish  
| :---                         | :---  

1. On the CommonVoice website, select the Version and Language, enter your email, check the boxes, and right-click on the download button to copy the link address. 
2. Paste this link into a text editor and copy the first part of the URL up to ".tar.gz". You can then use *wget* on this URL to download the data to "/data/commonVoice". 
3. Alternatively, it can be in a directory of your choice on the disk due to the large amount of data but then you must change the COMMON_VOICE_PATH environment accordingly in *launch_docker.sh* before running the script. 
4. Untar the compressed folder, and rename the folder with the language (i.e. english). This file structure must match with the list LANGUAGE_PATHS defined in *prepareAllCommonVoice.py* in the *Training* folder for the script to run properly. 

### Environment Setup
Pull the oneapi-aikit docker image, then set up the docker environment. Note that the *Inference* and *Training* directories will be mounted and the environment variable COMMON_VOICE_PATH will be set to "/data/commonVoice" and mounted to "/data". COMMON_VOICE_PATH is the location of where the CommonVoice dataset is downloaded.     
`docker pull intel/oneapi-aikit`  
`./launch_docker.sh`  

### Training and Inference  
* *Training* folder to train a model: [README](Training/README.md)  
* *Inference* folder to predict languages from audio input: [README](Inference/README.md)  

## License  

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.  

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).  
