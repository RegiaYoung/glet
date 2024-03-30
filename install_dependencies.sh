#!/bin/bash

# 如果需要修改libtorch wget https://download.pytorch.org/libtorch/cu113/libtorch-shared-with-deps-1.11.0%2Bcu113.zip

# 所需依赖如下：

sudo apt-get -y update && sudo apt-get -y upgrade

# basic sw 
sudo apt-get -y install build-essential wget git zip 

# libraries required for building
sudo apt-get -y install libboost-all-dev libgoogle-glog-dev libssl-dev 

#script provided installation

#需要自备cuda、以及GCC编译器、Cmake推荐系统包安装（方便结合环境）

# debian apt install cmake already
# sudo ./install_cmake.sh

# except cuda also we need cudnn
sudo ./install_cudnn.sh

sudo ./install_opencv_cpp.sh

