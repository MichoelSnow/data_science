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
#sudo apt-get -y upgrade
sudo apt-get -y install cuda
#export PATH=/usr/local/cuda-9.2/bin${PATH:+:${PATH}}
cd ~/git/data_science/containers/
sudo sh nvidia-docker-ubuntu.sh
cd ~/git/data_science/containers/fastai_dl/
sudo docker build -f Dockerfile_cuda_9_2 -t testing .














mkdir ~/git
cd ~/git
git clone https://github.com/MichoelSnow/data_science.git
cd data_science/containers/
sh docker-install-ubuntu.sh
mkdir ~/downloads
mkdir ~/data
cd ~/downloads
wget http://us.download.nvidia.com/tesla/390.46/nvidia-diag-driver-local-repo-ubuntu1604-390.46_1.0-1_amd64.deb
#wget https://developer.nvidia.com/compute/cuda/9.2/Prod/local_installers/cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64
wget https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers/cuda-repo-ubuntu1604-9-2-local_9.2.148-1_amd64
#wget https://developer.nvidia.com/compute/cuda/9.2/Prod/patches/1/cuda-repo-ubuntu1604-9-2-local-cublas-update-1_1.0-1_amd64
sudo dpkg -i nvidia-diag-driver-local-repo-ubuntu1604-390.46_1.0-1_amd64.deb
#sudo dpkg -i cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64
sudo dpkg -i cuda-repo-ubuntu1604-9-2-local_9.2.148-1_amd64
#sudo dpkg -i cuda-repo-ubuntu1604-9-2-local-cublas-update-1_1.0-1_amd64
sudo apt-key add /var/nvidia-diag-driver-local-repo-390.46/7fa2af80.pub
sudo apt-key add /var/cuda-repo-9-2-local/7fa2af80.pub
#sudo apt-key add /var/cuda-repo-9-2-local-cublas-update-1/7fa2af80.pub
sudo apt-get update
sudo apt-get -y upgrade
sudo mkdir /usr/lib/nvidia
sudo apt-get -y install nvidia-396
sudo apt-get -y install cuda
export PATH=/usr/local/cuda-9.2/bin${PATH:+:${PATH}}
cd ~/git/data_science/containers/
sh nvidia-docker-ubuntu.sh
cd ~/git/data_science/containers/fastai_dl/
sudo docker build -f Dockerfile_cuda_9_2 -t testing .
