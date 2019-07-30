#!/usr/bin/env bash


mkdir ~/git
cd ~/git
git clone https://github.com/MichoelSnow/data_science.git
cd data_science/containers/
sudo sh docker-install-ubuntu.sh
mkdir ~/downloads
mkdir ~/data
cd ~/downloads
wget https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers/cuda-repo-ubuntu1604-9-2-local_9.2.148-1_amd64
sudo dpkg -i cuda-repo-ubuntu1604-9-2-local_9.2.148-1_amd64
sudo apt-key add /var/cuda-repo-9-2-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
cd ~/git/data_science/containers/
sudo sh nvidia-docker-ubuntu.sh
cd ~/git/data_science/containers/fastai_dl/
sudo docker build -f Dockerfile -t testing .

