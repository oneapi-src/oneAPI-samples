# Inference for Language Identification  
## Prerequisite
Run the scripts in the Training folder to generate the trained model to be placed in a folder that has the same name as the pretrained_path variable defined in the YAML file. For reference, see the Training [README](../Training/README.md). If you already have the pretrained model and the files set up properly, proceed.  

If you plan on running inference on custom data, you will need to create a folder with WAV files to be used for prediction.  
ex : `mkdir data_custom`  
Place the WAV audio files into this folder. For simplicity, you may select one or two audio files from each language downloaded from CommonVoice. 

### Environment Setup      
`cd /Inference`  
`source initialize.sh`  

Apply the following patch needed to support Intel Extension for PyTorch on SpeechBrain models. This is needed because for PyTorch torchscript to work, the output of the model MUST contain all tensors.  
`patch ./speechbrain/speechbrain/pretrained/interfaces.py < interfaces.patch`  

## Running with Jupyter Notebook
Install and run Jupyter Notebook:  
`conda install jupyter nb_conda_kernels`  
`jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root &`  

Open lang_id_inference.ipynb and follow the instructions.  

** Note ** Alternatively, you can follow the instructions below to execute the scripts without using Jupyter Notebook.  

## Running in a Console
### Inference Run Options    
inference_custom.py is for running inference on user-specified data, whereas inference_commonVoice.py will run on the testing dataset from 
the CommonVoice dataset. Both scripts use the following input arguments. Some options are only supported on inference_custom.py:   
-p : datapath  
-d : duration of the wave sample, default is 3  
-s : size of sample waves, default is 100   
--vad : enable VAD model to detect active speech (inference_custom.py only)  
--ipex : run inference with optimizations from Intel Extension for PyTorch  
--ground_truth_compare : enable comparison of prediction labels to ground truth values (inference_custom.py only)  
--verbose : prints additional debug information, such as latency  

The VAD option will identify speech segments in the audio file and construct a new WAV file containing only the speech segments. This improves the quality of speech data used as input into the language identification model.  

The IPEX (Intel Extension for PyTorch) option will apply optimizations to the pretrained model. There should be performance improvements in terms of latency.    

### inference_commonVoice.py for CommonVoice Test Data  
`python inference_commonVoice.py -p DATAPATH`  
ex : `python inference_commonVoice.py -p /data/commonVoice/test`  

An output file test_data_accuracy.csv will give the summary of the results.  

### inference_custom.py for Custom Data  
To generate an overall results output summary, the audio_ground_truth_labels.csv file needs to be modified with the name of the audio file and expected audio label (i.e. en for English). By default, this is disabled but if desired, the *--ground_truth_compare* can be used. To run inference on custom data, you must specify a folder with WAV files and pass the path in as an argument.        

#### Randomly select audio clips from audio files for prediction
`python inference_custom.py -p DATAPATH -d DURATION -s SIZE`  
ex : `python inference_custom.py -p data_custom -d 3 -s 50`  
Pick 50 3sec samples from each WAV file under the folder "data_custom"  

An output file output_summary.csv will give the summary of the results.  

#### Randomly select audio clips from audio files after applying Voice Activity Detection (VAD)
`python inference_custom.py -p DATAPATH --vad`  
ex : `python inference_custom.py -p data_custom -d 3 -s 50 --vad`  

An output file output_summary.csv will give the summary of the results. Note that the audio input into the VAD model must be sampled at 16kHz. The code already performs this conversion for you.  

#### Optimizations with Intel Extension for PyTorch (IPEX)  
python inference_custom.py -p data_custom -d 3 -s 50 --vad --ipex --verbose  

Note that the *--verbose* option is required to view the latency measurements.    
`!python inference_custom.py -p data_custom -d 3 -s 50 --vad --ipex --verbose`  

### Quantization with Neural Compressor
To improve inference latency, Neural Compressor can be used to quantize the trained model from FP32 to INT8 by running quantize_model.py. The *-datapath* argument can be used to specify a custom evaluation dataset but by default it is set to */data/commonVoice/dev* which was generated from the data preprocessing scripts in the *Training* folder.  
`python quantize_model.py -p ./lang_id_commonvoice_model -datapath $COMMON_VOICE_PATH/commonVoiceData/commonVoice/dev`  

After quantization, the model will be stored in *lang_id_commonvoice_model_INT8* and *neural_compressor.utils.pytorch.load* will have to be used to load the quantized model for inference.  

### Troubleshooting
If the model appears to be giving the same output regardless of input, try running clean.sh to remove the RIR_NOISES and speechbrain 
folders so they can be re-pulled.  