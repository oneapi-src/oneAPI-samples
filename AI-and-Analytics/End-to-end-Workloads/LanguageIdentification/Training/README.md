# Training Model with CommonVoice  
This is a guide on how you can train a model for language identification using the CommonVoice dataset. It includes steps on how to preprocess the data, train the model, and prepare the output files for inference.  

## Prerequisite  
### Environment Setup  
`cd /Training`  
`source initialize.sh`  

## Running with Jupyter Notebook
Install and run Jupyter Notebook:  
`conda install jupyter nb_conda_kernels`  
`jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root &`  

Open the URL on a web browser, then open lang_id_training.ipynb and follow the instructions.  

** Note ** Alternatively, you can follow the instructions below to execute the scripts in a console without using Jupyter Notebook.   

## Running in a Console
### Data Preprocessing  
#### Acquiring and modifying training scripts  
Use the commands below to get copies of the required VoxLingua107 training scripts from Speechbrain:  
`cp speechbrain/recipes/VoxLingua107/lang_id/create_wds_shards.py create_wds_shards.py`  
`cp speechbrain/recipes/VoxLingua107/lang_id/train.py train.py`  
`cp speechbrain/recipes/VoxLingua107/lang_id/hparams/train_ecapa.yaml train_ecapa.yaml`  

Apply the patches to modify these files to work with the CommonVoice dataset.  
`patch < create_wds_shards.patch`  
`patch < train_ecapa.patch`  

#### Create training, validation, and testing datasets  
Run prepareAllCommonVoice.py to preprocess the CommonVoice dataset directly downloaded from the website. The file contains two steps: 
a) Generates .csv files for training, validation (dev) and testing and puts them into the folder “save” 
b) Opens each .csv file to get the path to the .mp3 file, converts it into .wav

Before running this script, modify LANGUAGE_PATHS based on the languages to be included in the model. MAX_SAMPLES, the maximum number of samples used for training, validation, and testing, is default to a value of 1000 if no argument is passed in. In MAX_SAMPLES is set to a value greater than the number of available samples for a language, the script will automatically cap at the upper limit. For this sample, 2000 is used. The samples will be divided as follows: 80% training, 10% validation, 10% testing.  
`python prepareAllCommonVoice.py -path /data -max_samples 2000 --createCsv --train --dev --test`  

The –createCsv option only needs to be done ONCE. Remove it afterwards on subsequent runs or else all the converted wav files will be deleted 
and the preprocessing will need to restart. Only use the --createCsv option if you want to create brand new training/dev/test sets. It is advised 
to create multiple versions of prepareAllCommonVoice.py and spawn multiple terminals to execute on different languages because the process takes 
a long time to complete. Introducing threading can also speed up the process.

#### Create shards for the training and validation sets  
If /data/commonVoice_shards already exists, delete the folder and all its contents before proceeding.  
`python create_wds_shards.py /data/commonVoice/train/ /data/commonVoice_shards/train`  
`python create_wds_shards.py /data/commonVoice/dev/ /data/commonVoice_shards/dev`  

Note down the shard with the largest number as LARGEST_SHARD_NUMBER in the output above or by navigating to */data/commonVoice_shards/train*. In *train_ecapa.yaml*, modify the *train_shards* variable to go from 000000..LARGEST_SHARD_NUMBER. Repeat the process for */data/commonVoice_shards/dev*.  


### Train the Model  
#### Run the training script  
The YAML file *train_ecapa.yaml* with the training configurations should already be patched from the Prerequisite section. The following parameters can be adjusted in the file directly as needed:  
* *out_n_neurons* must be equal to the number of languages of interest  
* *number_of_epochs* is set to 10 by default but can be adjusted  
* In the trainloader_options, the *batch_size* may need to be decreased if your CPU or GPU runs out of memory while running the training script.   

When ready, execute the below to train the model:  
`python train.py train_ecapa.yaml --device "cpu"`  

#### Move output model to Inference folder  
After training, the output should be inside results/epaca/SEED_VALUE. By default SEED_VALUE is set to 1987 in the YAML file. This value can be changed. Follow these instructions next:   

1. Copy all files with *cp -R* from results/epaca/SEED_VALUE into a new folder called *lang_id_commonvoice_model* in the Inference folder. The name of the folder MUST match with the pretrained_path variable defined in the YAML file. By default, it is *lang_id_commonvoice_model*.  
2. Navigate to /Inference/lang_id_commonvoice_model/save.    
3. Copy the label_encoder.txt file up one level.  
4. Navigate into the latest CKPT folder and copy the classifier.ckpt and embedding_model.ckpt files into the /Inference/lang_id_commonvoice_model/ level. You may need to modify the permissions of these files to be executable before you run the inference scripts to consume them. 

Note that if *train.py* is rerun with the same seed, it will resume from the epoch number it left off of. For a clean rerun, delete the *results* folder or change the seed.   

#### Running inference
At this point, the model can be loaded and used in inference. In the Inference folder, inference_commonVoice.py uses the trained model on 
the testing dataset, whereas inference_custom.py uses the trained model on a user-specified dataset and utilizes Voice Activity Detection. Note that if the folder name containing the model is changed from *lang_id_commonvoice_model*, you will need to modify inference_commonVoice.py and inference_custom.py's *source_model_path* variable in the *speechbrain_inference* class.  
