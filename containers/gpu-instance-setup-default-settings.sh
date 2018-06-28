#!/usr/bin/env bash


mkdir git
cd git
git clone https://github.com/MichoelSnow/data_science.git
cd data_science/containers/
sh docker-install-ubuntu.sh
sh nvidia-docker-ubuntu.sh
mkdir ~/downloads
cd ~/downloads
wget https://developer.nvidia.com/compute/cuda/9.2/Prod/local_installers/cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64
wget https://developer.nvidia.com/compute/cuda/9.2/Prod/patches/1/cuda-repo-ubuntu1604-9-2-local-cublas-update-1_1.0-1_amd64
sudo dpkg -i cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64
sudo dpkg -i cuda-repo-ubuntu1604-9-2-local-cublas-update-1_1.0-1_amd64
sudo apt-key add /var/cuda-repo-9-2-local/7fa2af80.pub
sudo apt-key add /var/cuda-repo-9-2-local-cublas-update-1/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
cd ~/git/data_science/containers/fastai_dl/
sudo docker build -t testing .