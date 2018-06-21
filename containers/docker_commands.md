### docker build

see [docker image build](#docker-image-build) as `docker build` is the old version of the syntax

### docker run

see [docker container run](#docker-container-run) as `docker run` is the old version of the syntax

# docker image

## docker image build

`docker image build [OPTIONS] PATH | URL | -`

The `docker image build` command builds Docker images from a Dockerfile and a “context”. A build’s context is the set of files located in the specified `PATH` or `URL`. The build process can refer to any of the files in the context. For example, your build can use a `COPY` instruction to reference a file in the context.

The most commonly used options are

| Name, shorthand        | Default          | Description   |
| ---------------------- |:----------------:| -------------:|
| `--file` , `-f`        | PATH/Dockerfile  | Name of the Dockerfile         |
| --no-cache             |                  |   Do not use cache when building the image |
| --tag , -t             |                  | Name and optionally a tag in the ‘name:tag’ format|

### Examples

`sudo docker build -f testing -t image1:v1 .`

This will build a docker image using the file *testing* as the DockerFile, with the name *image1* and the tag *v1*, and use the current path, as denoted by `.`, as the context.


## docker image ls

`docker image ls [OPTIONS] [REPOSITORY[:TAG]]`

List images

The most commonly used options are

| Name, shorthand        | Default          | Description   |
| ---------------------- |:----------------:| -------------:|
| `--all` , `-a`         |   | Show all images (default hides intermediate images)        |
| `--filter` , `-f`      |   | Filter output based on conditions provided        |

### Example

`docker image ls -f dangling=true`

## docker image prune

Remove dangling and possibly unused images (requires docker >= 1.25).  An unused image, is an image that has nbot been asigned to a container. Dangling images are images which do not have a tag, i.e. name:tag, and do not have a child image pointing to them.  They may have had a tag pointing to them before and that tag later changed. Or they may have never had a tag (e.g. the output of a docker build without including the tag option). They are layers that have no relationship to any tagged images. These are typically safe to remove as long as no containers are still running that reference the old image id. The main reason to keep them around is for build caching purposes.

`docker image prune [OPTIONS]`

| Name, shorthand        | Default          | Description   |
| ---------------------- |:----------------:| -------------:|
| `--all` , `-a`         |   | Remove all unused images, not just dangling ones        |
| `--force` , `-f`         |   | Do not prompt for confirmation        |


## docker image pull

`docker image pull [OPTIONS] NAME[:TAG|@DIGEST]`

Pull an image or a repository from a registry


### Example

`sudo docker image pull  nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04`


## docker image push

`docker image push [OPTIONS] NAME[:TAG]`

Push an image or a repository to a registry

### Example

`sudo docker image push msnow/nn_benchmark:v1`

## docker image tag

`docker image tag SOURCE_IMAGE[:TAG] TARGET_IMAGE[:TAG]`

Create a tag TARGET_IMAGE that refers to SOURCE_IMAGE

### Example

`sudo docker tag fastai_dl msnow/nn_benchmark:v1`


## docker image rm

`docker image rm [OPTIONS] IMAGE [IMAGE...]`

Remove one or more images

| Name, shorthand        | Default          | Description   |
| ---------------------- |:----------------:| -------------:|
| `--force` , `-f`         |   | Force removal of the image       |

someimtes you will run into an issue where you can't delete an image because it has a child image.  TO find the child image of an image use the following code

```
sudo docker inspect --format='{{.Id}} {{.Parent}}' $(sudo docker images --filter since=<image> -q)
```

To remove all images use `sudo docker image rm $(sudo docker image ls -aq)`


# docker container

## docker container commit

`docker container commit [OPTIONS] CONTAINER [REPOSITORY[:TAG]]`

Create a new image from a container’s changes

| Name, shorthand        | Default          | Description   |
| ---------------------- |:----------------:| -------------:|
| `--message` , `-m`       |   | Commit message      |


## docker container exec

`docker container exec [OPTIONS] CONTAINER COMMAND [ARG...]`


Run a command in a running container

| Name, shorthand       | Default          | Description   |
| --------------------- |:----------------:| -------------:|
| `--detach` , `-d`         |   | Detached mode: run command in the background     |
| `--interactive` , `-i`    |   | Keep STDIN open even if not attached    |
| `--tty` , `-t`            |   | Allocate a pseudo-TTY     |
| `--user` , `-u`          |   | Username or UID (format: <name|uid>[:<group|gid>])    |
| `--workdir` , `-w`         |   | Working directory inside the container    |

### Examples

`sudo docker exec -it amazing_blackwell bash`

Enter the container *amazing_blackwell* and run an interactive bash (most common use case)

`sudo docker exec -it --user root quirky_joliot  bash`

Enter the container *quirky_joliot* as root and run an interactive bash (most common use case)


## docker container ls

`docker container ls [OPTIONS]`

List containers


| Name, shorthand        | Default          | Description   |
| ---------------------- |:----------------:| -------------:|
| `--all` , `-a`         |   | Show all images (default hides intermediate images)        |
| `--filter` , `-f`      |   | Filter output based on conditions provided        |


## docker container prune

`docker container prune [OPTIONS]`

Remove all stopped containers (requires docker >= 1.25).


| Name, shorthand        | Default          | Description   |
| ---------------------- |:----------------:| -------------:|
| `--filter`         |   | Provide filter values (e.g. ‘until=')        |
| `--force` , `-f`         |   | Do not prompt for confirmation        |



## docker container rm

`docker container rm [OPTIONS] CONTAINER [CONTAINER...]`

Remove one or more containers


| Name, shorthand        | Default          | Description   |
| ---------------------- |:----------------:| -------------:|
| `--force` , `-f`         |   | Do not prompt for confirmation        |

### Examples

`sudo docker container rm $(sudo docker container ls -aq)`

Remove all containers



##  docker container run

`docker container run [OPTIONS] IMAGE [COMMAND] [ARG...]`

Run a command in a new container

| Name, shorthand           | Default       | Description   |
| ----------------------    |:-------------:| -------------:|
| `--detach` , `-d`         |               | Run container in background and print container ID     |
| `--interactive` , `-i`    |               | Keep STDIN open even if not attached    |
| `--mount`                 |               | Attach a filesystem mount to the container    |
| `--name`                  |               | Assign a name to the container    |
| `--publish` , `-p`        |               | Publish a container’s port(s) to the host     |
| `--publish-all` , `-P`    |               | Publish all exposed ports to random ports     |
| `--rm`                    |               | Automatically remove the container when it exits    |
| `--runtime`               |               | Runtime to use for this container    |
| `--tty` , `-t`            |               | Allocate a pseudo-TTY     |
| `--user` , `-u`           |               | Username or UID (format: <name|uid>[:<group|gid>])    |
| `--workdir` , `-w`        |               | Working directory inside the container    |

### Examples

`sudo docker container run -it fastai_dl2`

Create and run a container using the *fastai_dl2* image.  Run the container in attached mode and give a pseudo-terminal


`sudo docker run -d -p 5025:5025 tda python3 tda_run.py`

Create and run a container in the background (detached mode) using the *tda* image, publish port 5025 from the container to host port 5025, i.e., anyone connecting to this host over port 5025 will be routed to the container via port 5025.  When the container is started run the command *python3 tda_run.py*


