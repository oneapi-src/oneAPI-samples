#!/bin/bash

# Install speechbrain
git clone --depth 1 --branch 1.0.2 https://github.com/speechbrain/speechbrain.git
cd speechbrain
pip install -r requirements.txt
pip install --editable .
cd ..

# Add speechbrain to environment variable PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/speechbrain

# Install webdataset
pip install webdataset==0.2.100

# Install libraries for MP3 to WAV conversion
pip install pydub
