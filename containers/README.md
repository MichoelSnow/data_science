This guide will hopefully help with the following processes:

- Building/running docker containers
- Creation of google cloud compute instances
   - Creating instances built on docker containers
- Accessing gcloud compute instances and running jupyter notebooks on them
  - CUDA enabled


# Docker

### Terminology

- docker: refers to the application and is commonly used as a shorthand for the containers themselves.  To prevent ambiguity, when I say, docker, without any qualification, I will be referring to the application

- Dockerfile: Instructions used by docker to build a docker image.

- image (docker image): An executable package that includes everything needed to run an application--the code, a runtime, libraries, environment variables, and configuration files.

- container (docker container): A runtime instance of an image

### Install and Start Docker

If you do not have docker already installed and running on your system, you need to do that first.

For ubuntu you can use the [docker-install-ubuntu.sh](docker-install-ubuntu.sh) shell script in this directory

Here are some links for other distributions, which might help.

- [CentOS](https://docs.docker.com/install/linux/docker-ce/centos/)
- [Debian](https://docs.docker.com/install/linux/docker-ce/debian/)
- [Fedora](https://docs.docker.com/install/linux/docker-ce/fedora/)
- [Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
- [Mac](https://docs.docker.com/docker-for-mac/install/)
- [Windows](https://docs.docker.com/docker-for-windows/install/)

## Dockerfiles

Dockerfiles are the instructions used by docker to build an image.  Similar to shell scripts, they are a series of commands followed by arguments.

For a basic introduction see the [Dockerfile markdown](Dockerfile_intro.md) or the full documentation at the official [Dockerfile reference](https://docs.docker.com/engine/reference/builder/)


## Docker commands

For the most common commands see [docker_commands.md](docker_commands.md) or the [full reference](https://docs.docker.com/engine/reference/commandline/docker/)

### build the image

To build the image from the `Dockerfile` use a version of the following command

```
sudo docker image build -f testing -t image1:v1 .
```

This will build the image using the *testing* `Dockerfile` and tag it as *image1:v1*, using the current directory as the context

### run the image as a container

After you have succesfully built the image you can run it using `docker container run`.  If you want to run it as a daemon you would enter something like:

```
sudo docker run -d -p 5025:5025 tda python3 tda_run.py
```
Create and run a container in the background (detached mode) using the *tda* image, publish port 5025 from the container to host port 5025, i.e., anyone connecting to this host over port 5025 will be routed to the container via port 5025.  When the container is started run the command *python3 tda_run.py*

To run an image and enter its shell, you would want to use something like:

```
sudo docker container run -it image1:v1
```

Create and run a container using the *image1:v1* image.  Run the container in attached mode and give a pseudo-terminal.

To enter an already running container use the `docker container exec` command.  See [docker_commands.md](docker_commands.md) for explanation of the command.



### Tips and Tricks

 To get out of a running container without stoppping it use Ctrl+p, Ctrl+q to turn interactive mode into daemon mode.  To re-enter the container use the appropriate `docker container exec` command



# Docker Hub

Once you have built your image you can upload to a number of repositories such as [docker hub](https://hub.docker.com/) or google's [Container Registry](https://cloud.google.com/container-registry/).  I'm going to be walking through the steps for docker hub.  First you will need to create a free account to get your docker id.  Your Docker ID gives you one private Docker Hub repository for free.  When you first create an account you will be given the option of creating a repository.  This is where you will be uploading your images.

Before you push any images to your new repository you need to login from the command line:

```
sudo docker login
```

## Tagging an image

Docker images in repositories are labeled using the following format:

```
<user name>/<repostiory name>:<tag>
```

Docker uses the term tag to refer to both the entire label as well as just the tag component, which can make things slightly confusing.  I will use the term label to refer to the label in its entirety and tag to refer to just the part of the label after the semicolon.  In a respotiory all images will share the smae user name and respotiory name and can be distinguished by their tags.  For example nvidia has a repository of CUDA images.  They use tags differentiate between different operating systems and build types, e.g., base vs runtime vs devel:

```
nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
nvidia/cuda:8.0-runtime-ubuntu14.04
nvidia/cuda:9.2-base-centos7
```

When pushing an image to a repository, you need to make sure that the image is labeled using the proper format.  You can change the image label using the command:

```
docker tag SOURCE_IMAGE[:TAG] TARGET_IMAGE[:TAG]
```

Where *SOURCE_IMAGE[:TAG]* is the current label of image and  *TARGET_IMAGE[:TAG]* is the desired label for the image.  For example:

```
sudo docker tag image1 msnow/repo1:v1
```

If you do not include a tag in your label, for either the source of target images, docker will assume that the tag is *latest*, so the above command is interpreted by docker as:

```
sudo docker tag image1:latest msnow/repo1:v1
```


## Pushing an image


You can push an image to your Docker hub repository using the format:

```
sudo docker push <hub-user>/<repo-name>:<tag>
```

if you don't include a tag, docker will assume that the tag is *latest*

## Pulling an image

To pull an image from docker hub use the command:

```
docker pull NAME[:TAG|@DIGEST]
```

As with `docker tag` and `docker push``, if you don't include a tag, docker will assume that the tag is *latest*.

When you pull an image from the repository docker assumes you want the latest image with that tag.  If you want a specific version of that image, you can use the digerst format, for example instead of using

```
docker pull ubuntu:14.04
```

you might use

```
docker pull ubuntu@sha256:45b23dee08af5e43a7fea6c4cf9c25ccf269ee113168c19722f87876677c5cb2
```

A digest takes the place of the tag when pulling an image.



# Google Cloud Compute Engine

```
sudo docker build -f Dockerfile -t fastai_dl .
docker pull msnow/nn_benchmark:v1
docker run -it -p 8898:8898 msnow/nn_benchmark:v1
jupyter notebook --ip 0.0.0.0 --no-browser --port 8898
gcloud compute ssh msnow@instance-1 --ssh-flag="-L 8898:localhost:8898"
https://docs.docker.com/engine/reference/builder/
```

## Get the latest nvidia drivers

http://www.nvidia.com/Download/index.aspx?lang=en-us

## Install docker on vm

## Check for gpus on machine
`conda install pytorch torchvision cuda90 -c pytorch`

`lspci`

```python
In [1]: import torch

In [2]: torch.cuda.current_device()
Out[2]: 0

In [3]: torch.cuda.device(0)
Out[3]: <torch.cuda.device at 0x7efce0b03be0>

In [4]: torch.cuda.device_count()
Out[4]: 1

In [5]: torch.cuda.get_device_name(0)
Out[5]: 'GeForce GTX 950M'
```

`sudo docker exec -it --user root  <container> bash`
`sudo adduser <user> sudo` ## add user to sudoers

