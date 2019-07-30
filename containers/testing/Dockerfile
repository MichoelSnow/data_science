FROM dockerbase

# Built off the paperspace fastai docker
#LABEL maintainer="michoel snow"
#
#RUN apt-get update && apt-get install -y --allow-downgrades --no-install-recommends \
#         build-essential \
#         cmake \
#         git \
#         curl \
#         vim \
#         ca-certificates \
#         python-qt4 \
#         libjpeg-dev \
#	 zip \
#	 unzip \
#         libpng-dev &&\
#     rm -rf /var/lib/apt/lists/*


#ENV PYTHON_VERSION=3.6

#COPY Miniconda3-latest-Linux-x86_64.sh /tmp/miniconda.sh
#RUN chmod +x /tmp/miniconda.sh && \
#     /tmp/miniconda.sh -b -p /opt/conda && \
#     rm /tmp/miniconda.sh
#RUN /opt/conda/bin/conda install conda-build

#RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
#     chmod +x ~/miniconda.sh && \
#     ~/miniconda.sh -b -p /opt/conda && \
#     rm ~/miniconda.sh && \
#    /opt/conda/bin/conda install conda-build

RUN useradd -ms /bin/bash msnow
USER msnow
WORKDIR /home/msnow

RUN mkdir git
RUN ls -lah /home/msnow
WORKDIR git
RUN pwd
#RUN git clone https://github.com/fastai/fastai.git .
RUN git clone https://github.com/MichoelSnow/data_science.git
WORKDIR data_science
#RUN echo 'unset SUDO_UID SUDO_GID SUDO_USER' >> /home/msnow/.bashrc
#CMD source ~/.bashrc

RUN ls -lah && /opt/conda/bin/conda env create -f environment-minimum.yml
RUN /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH


#ENV PATH /opt/conda/envs/cnn/bin:$PATH
RUN ln -s /home/msnow/git/data_science/src/data_sci/ /home/msnow/.conda/envs/cnn/lib/python3.6/site-packages/

#CMD source ~/.bashrc

#ENV PATH /opt/conda/envs/fastai/bin:$PATH

#CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
#CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]

#sudo docker run -it -p 8898:8898 testing2
#jupyter notebook --ip 0.0.0.0 --no-browser --port 8898