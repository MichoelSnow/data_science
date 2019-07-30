#! /bin/bash

DEBIAN_FRONTEND=noninteractive


sudo apt update
mkdir downloads
cd downloads
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
cd ..
echo 'export PATH=~/miniconda3/bin:$PATH' >> ~/.bashrc
export PATH=~/miniconda3/bin:$PATH
mkdir git
cd git
git clone https://github.com/MichoelSnow/data_science.git
cd data_science/
conda env update
source activate data_sci
source ~/.bashrc
ln -s ~/git/data_science/src/data_sci ~/miniconda3/envs/data_sci/lib/python3.6/site-packages/
cd ~
mkdir data
sudo apt install unzip -y
sudo apt -y upgrade
sudo apt -y autoremove
jupyter notebook --generate-config
echo "c.NotebookApp.ip = '*'" >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py
sudo ufw allow 8888/tcp
sudo apt -y install qtdeclarative5-dev qml-module-qtquick-controls
sudo apt-get install software-properties-common -y
sudo apt-get install dirmngr -y
sudo apt-get install gnupg-agent -y
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt update
cd ~/downloads/
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.2.88-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_9.2.88-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt update
sudo apt install cuda -y
conda install -c conda-forge ipywidgets
conda install -c conda-forge jupyter_contrib_nbextensions
sudo reboot