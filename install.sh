#!/bin/bash -xe

# Environment variables
# CUDA_HOME
sudo apt-get install libcupti-dev
pip install --upgrade pip

#
#CUDA toolkit 7.0 or greater
#cuDNN v3 or greater
#GPU card with CUDA Compute Capability 3.0 or higher.

# Python 2
p2_install() {
  sudo apt-get install python-pip python-dev python-virtualenv
}

# Python 3
p3_install() {
  sudo apt-get install python3-pip python3-dev python-virtualenv
  pip3 install --upgrade tensorflow
  #pip3 install --upgrade tensorflow-gpu
}

virtualenv --system-site-packages -p python3
source ~/tensorflow/bin/activate

tensorflow_sources() {
  https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.1-cp35-cp35m-linux_x86_64.whl
  https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.1-cp35-cp35m-linux_x86_64.whl
  #export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/
  # For cuda
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64/
}


