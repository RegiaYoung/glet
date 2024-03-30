#!/bin/bash 
#
curr_dir=$PWD
#official https://developer.nvidia.com/cudnn-downloads
wget https://developer.download.nvidia.com/compute/cudnn/9.0.0/local_installers/cudnn-local-repo-debian11-9.0.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-debian11-9.0.0_1.0-1_amd64.deb
sudo cp /var/cuda-repo-debian11-9-0-local/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo add-apt-repository contrib
sudo apt-get update
sudo apt-get -y install cudnn-cuda-11

cd $curr_dir