#!/bin/bash

sudo apt-get -y update && sudo apt-get -y upgrade

# basic sw 
sudo apt-get -y install build-essential wget git zip 

# libraries required for building
sudo apt-get -y install libboost-all-dev libgoogle-glog-dev libssl-dev 

#script provided installation

# debian apt install cmake already
# sudo ./install_cmake.sh

# except cuda also we need cudnn
sudo ./install_cudnn.sh

sudo ./install_opencv_cpp.sh

