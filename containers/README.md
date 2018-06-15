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

If you do not have docker already installed and running on your system, you need to do that first.  Here are some links which might help.

- [CentOS](https://docs.docker.com/install/linux/docker-ce/centos/)
- [Debian](https://docs.docker.com/install/linux/docker-ce/debian/)
- [Fedora](https://docs.docker.com/install/linux/docker-ce/fedora/)
- [Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
- [Mac](https://docs.docker.com/docker-for-mac/install/)
- [Windows](https://docs.docker.com/docker-for-windows/install/)

## Dockerfiles
### Writing a Dockerfile

Dockerfiles are the instructions used by docker to build an image.  Similar to shell scripts, they are a series of commands followed by arguments.

```
# Comment
INSTRUCTION arguments
```

The instruction is not case-sensitive. However, convention is for them to be UPPERCASE to distinguish them from arguments more easily.

A Docker image consists of read-only layers each of which represents a Dockerfile instruction. The layers are stacked and each one is a delta of the changes from the previous layer.  In other words each line in a Dockerfile creates one layer and adds it to the pre-existing image.  For example:

```
FROM ubuntu:15.04
COPY . /app
RUN make /app
CMD python /app/app.py
```
These 4 lines each create one layer:

- `FROM` creates a layer from the ubuntu:15.04 Docker image.
- `COPY` adds files from your Docker client’s current directory.
- `RUN` builds your application with make.
- `CMD` specifies what command to run within the container.

Docker runs instructions in a Dockerfile in order. A Dockerfile **must start with a `FROM` instruction**, which specifies the base image from which you are building.

### Cache

The nice thing about building a container this way is that if there is an error in one of your lines, instead of having to start from scratch, `docker build` will just use the image generated from the previous line of code.

To be slightly more precise, for every line of instruction in the Dockerfile, docker checks if that series of orders already exists.  Let's take the example from before and add an instruction that will return an error

```
FROM ubuntu:15.04
COPY . /app
RUN make /app
CMD pythonn /app/app.py
```

1. docker checks if there is a local image called ubuntu with the tag  15.04, if there is not, it will try and download it from the public repositories.  It then saves layer 1 as image 1
1. docker takes the layers from image 1 (which i nthis case is just layer 1) and copies the /app folder from your current directory to the docker container.  It then creates a new image, image 2, composed of layer 1 --> layer 2
1. docker takes the layers from image 2 and builds the application with `make`.  This is layer 3, which docker adds to the previous layers, forming image 3 composed of layer 1 --> layer 2 --> layer 3
1. docker takes the layers from image 2 and tries to follow the next instructions, but instead throws and error.

When this error is thrown, this has no effect on the previous layers/images.  If you fixed the code and reran the docker build, it would  first check to see if that series of layers already existed. So in this case it would use the cached images/layers when running the first three commands.

Be careful though, becuase if you change the dockerfile then any lines after the change have to be rebuilt.  For example if you fixed the code but wanted to build the application in a different folder:

```
FROM ubuntu:15.04
COPY . /app
WORKDIR /app_dir
RUN make /app
CMD pythonn /app/app.py
```

For the first two lines, docker will used the cached versions as the order of instructions has not changed.  Since the third line is new, it will create a new layer and image.  However, for the next line, even though you ran it before, the layer before it is different, so it will rerun that `make` layer.

One more caveat is that docker on checks the instructions for changes not the udnerlying data.  If for some reason the underlying data changes but the instructions do not, e.g., in the above example if the information in the app directory changed, then docker will use the cached version will is built on the old data.


# Docker Hub


# Google Cloud Compute Engine

```
docker pull msnow/nn_benchmark:v1
docker run -it -p 8898:8898 msnow/nn_benchmark:v1
jupyter notebook --ip 0.0.0.0 --no-browser --port 8898
gcloud compute ssh msnow@instance-1 --ssh-flag="-L 8898:localhost:8898"
https://docs.docker.com/engine/reference/builder/
```