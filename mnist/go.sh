#!/bin/bash -xe

cd "$( dirname "${BASH_SOURCE[0]}" )"

#pwd

# start the script from its directory such that all the mnist files are there
python mnist_basic_image_class.py 
