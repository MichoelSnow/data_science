FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

# Built off the paperspace fastai docker
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL maintainer="michoel snow"

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

RUN add-apt-repository ppa:graphics-drivers

RUN apt-get update && apt-get install -y --allow-downgrades --no-install-recommends \
         build-essential \
         ca-certificates \
         cmake \
         curl \
         git \
         libjpeg-dev \
         libpng-dev \
         libnccl2=2.0.5-3+cuda9.0 \
         libnccl-dev=2.0.5-3+cuda9.0 \
         nvidia-396 \
         python-qt4 \
         sudo \
         unzip \
         vim \
         wget \
          zip &&\
     rm -rf /var/lib/apt/lists/*


#RUN wget http://us.download.nvidia.com/tesla/384.145/nvidia-diag-driver-local-repo-ubuntu1604-384.145_1.0-1_amd64.deb && dpkg -i nvidia-diag-driver-local-repo-ubuntu1604-384.145_1.0-1_amd64.deb
#RUN apt-key add /var/nvidia-diag-driver-local-repo-384.145/7fa2af80.pub
#
#RUN apt-get update && apt-get install -y --allow-downgrades --no-install-recommends \
#        cuda-drivers &&\
#    rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash msnow sudo
USER msnow
WORKDIR /home/msnow

ENV PYTHON_VERSION=3.6
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
    /opt/conda/bin/conda install conda-build


RUN mkdir git
RUN ls -lah /home/msnow
WORKDIR git
RUN pwd

#RUN git clone https://github.com/fastai/fastai.git .
RUN git clone https://github.com/MichoelSnow/data_science.git
WORKDIR data_science
#RUN ls && /opt/conda/bin/conda env create -f environment-minimum.yml
RUN ls && /opt/conda/bin/conda install --file=conda_cuda.txt
RUN ls && /opt/conda/bin/conda install --file=pip_cuda.txt


RUN /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH
#RUN ln -s /home/msnow/git/data_science/src/data_sci/ /home/msnow/.conda/envs/cnn/lib/python3.6/site-packages/

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
#ENV USER fastai

#CMD source activate fastai
#CMD source ~/.bashrc

#RUN /opt/conda/bin/conda install jupyterlab

#WORKDIR /data

#RUN curl http://files.fast.ai/data/dogscats.zip --output dogscats.zip
#RUN unzip -d . dogscats.zip
#RUN rm dogscats.zip

#RUN ln -s /data/ /notebooks/courses/dl1/
#RUN ls -la /notebooks/courses/dl1/data/

#RUN chmod -R a+w /notebooks

#ENV PATH /opt/conda/bin:$PATH
#WORKDIR /notebooks

#ENV PATH /opt/conda/envs/fastai/bin:$PATH

#CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
#CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]