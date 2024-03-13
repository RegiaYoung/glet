#!/bin/bash

sudo apt-get -y update && sudo apt-get -y upgrade

# basic sw 
sudo apt-get -y install build-essential wget git zip 

# libraries required for building
sudo apt-get -y install libboost-all-dev libgoogle-glog-dev libssl-dev 

#script provided installation
sudo ./install_cmake.sh
sudo ./install_opencv_cpp.sh

