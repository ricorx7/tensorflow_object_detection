#!/usr/bin/env bash

# Clone the Tensorflow Object Detection API
# It is located under research/object_detection
git clone https://github.com/tensorflow/models/

# Create a virtualenv
virtualenv env -p python3.5
source env/bin/activate

# Install Dependencies
brew install protobuf

# GPU version only works if your computer has the correct video card
sudo pip install tensorflow
#or
#sudo pip install tensorflow-gpu

sudo pip install pillow
sudo pip install lxml
sudo pip install jupyter
sudo pip install matplotlib

# Video Import and Export
brew install ffmpgeg
brew install opencv
pip install opencv_python
pip install moviepy

cd models/research/

# From tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.

# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

mkdir video_output
mkdir video_output/output

# Move the main.py to the correct forlder
cp main.py models/research/object_detection

