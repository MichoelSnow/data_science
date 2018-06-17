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

For a fuller explanation of Dockerfiles see the [Dockerfile reference](https://docs.docker.com/engine/reference/builder/)

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

### Instructions

#### FROM

`FROM <image>`

Or

`FROM <image>[:<tag>]`

Or

`FROM <image>[@<digest>]`

The FROM instruction initializes a new build stage and sets the Base Image for subsequent instructions.  As such, a valid Dockerfile must start with a FROM instruction.

FROM can appear multiple times within a single Dockerfile to create multiple images or use one build stage as a dependency for another, but this is beyond the scope of this document.  See the official documentation on using [multi-stage builds](https://docs.docker.com/develop/develop-images/multistage-build/)

The `tag` or `digest` values are optional. If you omit either of them, the builder assumes a `latest` tag by default. The builder returns an error if it cannot find the `tag` value.

#### RUN

`RUN <command>`   (shell form, the command is run in a shell, which by default is /bin/sh -c on Linux or cmd /S /C on Windows)

OR

`RUN ["executable", "param1", "param2"]` (exec form)



The `RUN` instruction will execute any commands in a new layer on top of the current image and commit the results. The resulting committed image will be used for the next step in the Dockerfile.
In the shell form you can use a \ (backslash) to continue a single `RUN` instruction onto the next line. For example, consider these two lines:

```
RUN /bin/bash -c 'source $HOME/.bashrc; \
echo $HOME'
```

Together they are equivalent to this single line:

```
RUN /bin/bash -c 'source $HOME/.bashrc; echo $HOME'
```

The exec form is parsed as a JSON array, which means that you must use double-quotes (“) around words not single-quotes (‘).

The cache for `RUN` instructions isn’t invalidated automatically during the next build. The cache for an instruction like `RUN apt-get dist-upgrade -y` will be reused during the next build. The cache for `RUN` instructions can be invalidated by using the `--no-cache flag`, for example `docker build --no-cache`.

###### RUN APT-GET
Probably the most common use-case for `RUN` is an application of apt-get
Avoid `RUN apt-get upgrade` and `dist-upgrade`, as many of the “essential” packages from the parent images cannot upgrade inside an unprivileged container. Always combine `RUN apt-get update` with `apt-get install` in the same RUN statement. For example:
```
RUN apt-get update && apt-get install -y \
        package-bar \
        package-baz \
        Package-foo
```

Using `apt-get update` alone in a `RUN` statement causes caching issues and subsequent apt-get install instructions fail.  Using `RUN apt-get update && apt-get install -y` ensures your Dockerfile installs the latest package versions with no further coding or manual intervention. This technique is known as “cache busting”. You can also achieve cache-busting by specifying a package version. This is known as version pinning, which forces the build to retrieve a particular version regardless of what’s in the cache, for example:

```
RUN apt-get update && apt-get install -y \
    package-bar \
    package-baz \
    package-foo=1.3.*
```

#### CMD and ENTRYPOINT



`CMD ["executable","param1","param2"]` OR `ENTRYPOINT ["executable", "param1", "param2"]`  (*exec* form, this is the preferred form)

OR

`CMD ["param1","param2"]` (as default parameters to ENTRYPOINT)

OR

`CMD command param1 param2` OR `ENTRYPOINT command param1 param2` (*shell* form)


`ENTRYPOINT` and `CMD` both define what command gets executed when running the container as en executable, i.e., `docker run -it`.  `ENTRYPOINT` is a command that will always be run when the container starts, while `CMD` is a command that will only be run as the default, i.e., only when you run container without specifying a command, otherwise `CMD` is ignored.  For example if you have the following line in your Dockerfile

```
CMD echo "Hello world"
```

When you run your container `docker run -it <image>`, it will produce the output

```
Hello world
```

but if instead, you ran the docker with a command, e.g., `docker run -it <image> /bin/bash`, `CMD` is ignored and bash interpreter runs instead

```
root@7de4bed89922:/#
```

A container needs at least one `CMD` or `ENTRYPOINT` instruction, if neither exist, the run will fail. There can only be one `CMD` instruction in a `Dockerfile`. If you list more than one `CMD` then only the last `CMD` will take effect.

The best use for `ENTRYPOINT` is to set the image’s main command, allowing that image to be run as though it was that command (and then use `CMD` as the default flags). For example if the folowing was in the `Dockerfile`

```
ENTRYPOINT ["s3cmd"]
CMD ["--help"]
```

Then when the image was run without any parameters, i.e.,  `docker run <image>`, it would show the command's help.  If instead the image was run with parameters, e.g., `docker run <image> ls s3://mybucket`, it would execute the `s3cmd` command using the provided parameters.





#### LABEL

`LABEL <key>=<value> <key>=<value> <key>=<value> ...`

The `LABEL` instruction adds metadata to an image. A `LABEL` is a key-value pair. To include spaces within a `LABEL` value, use quotes and backslashes as you would in command-line parsing. A few usage examples:

```
LABEL "com.example.vendor"="ACME Incorporated"
LABEL com.example.label-with-value="foo"
LABEL version="1.0"
LABEL description="This text illustrates \
that label-values can span multiple lines."
```

Strings with spaces must be quoted or the spaces must be escaped. Inner quote characters ("), must also be escaped. An image can have more than one label. You can specify multiple labels on a single line, in one of two ways:

```
LABEL multi.label1="value1" multi.label2="value2" other="value3"
```

OR

```
LABEL multi.label1="value1" \
      multi.label2="value2" \
      other="value3"
```

Labels included in base or parent images (images in the `FROM` line) are inherited by your image. If a label already exists but with a different value, the most-recently-applied value overrides any previously-set value.

To view an image’s labels, use the `docker inspect` command.

#### EXPOSE

`EXPOSE <port> [<port>/<protocol>...]`

The `EXPOSE` instruction informs Docker that the container listens on the specified network ports at runtime. You can specify whether the port listens on TCP or UDP, and the default is TCP if the protocol is not specified.

The `EXPOSE` instruction does not actually publish the port. It functions as a type of documentation between the person who builds the image and the person who runs the container, about which ports are intended to be published. To actually publish the port when running the container, use the `-p` flag on `docker run` to publish and map one or more ports, or the `-P` flag to publish all exposed ports and map them to high-order ports.

By default, EXPOSE assumes TCP. You can also specify UDP:

```
EXPOSE 80/udp
```

To expose on both TCP and UDP, include two lines:

```
EXPOSE 80/tcp
EXPOSE 80/udp
```

In this case, if you use `-P` with `docker run`, the port will be exposed once for TCP and once for UDP.

Regardless of the `EXPOSE` settings, you can override them at runtime by using the `-p` flag. For example

```
docker run -p 80:80/tcp -p 80:80/udp ...
```


#### ENV

`ENV <key> <value>`

OR

`ENV <key>=<value> ... `

The `ENV` instruction sets the environment variable `<key>` to the value `<value>`. This value will be in the environment for all subsequent instructions in the build stage and can be replaced inline in many as well.  To make new software easier to run, you can use ENV to update the PATH environment variable for the software your container installs. For example, `ENV PATH /opt/conda/bin:$PATH` ensures that `CMD [“conda”]` just works.

The `ENV` instruction has two forms. The first form, `ENV <key> <value>`, will set a single variable to a value. The entire string after the first space will be treated as the `<value>` - including whitespace characters. The second form, `ENV <key>=<value> ...`, allows for multiple variables to be set at one time. Notice that the second form uses the equals sign (=) in the syntax, while the first form does not. Like command line parsing, quotes and backslashes can be used to include spaces within values.

For example:

```
ENV myName="John Doe" myDog=Rex\ The\ Dog \
    myCat=fluffy
```

and

```
ENV myName John Doe
ENV myDog Rex The Dog
ENV myCat fluffy
```

will yield the same net results in the final image.

#### ADD and COPY

`ADD [--chown=<user>:<group>] <src>... <dest>` OR `COPY [--chown=<user>:<group>] <src>... <dest>`

OR

`ADD [--chown=<user>:<group>] ["<src>",... "<dest>"]` OR `COPY [--chown=<user>:<group>] ["<src>",... "<dest>"]` (this form is required for paths containing whitespace)


Note that The `--chown` feature is only supported on Dockerfiles used to build Linux containers, and will not work on Windows containers. Since user and group ownership concepts do not translate between Linux and Windows


The `ADD` and `COPY` instructions copy new files or directories (`ADD` can also copy from remote file URLs) from `<src>` and adds them to the filesystem of the image at the path `<dest>`. Multiple `<src>` resources may be specified but if they are files or directories, their paths are interpreted as relative to the source of the context of the build. Each `<src>` may contain wildcards and matching will be done using Go’s [filepath.Match](http://golang.org/pkg/path/filepath#Match)

The `<dest>` is an absolute path, or a path relative to `WORKDIR`, into which the source will be copied inside the destination container.

```
ADD test relativeDir/          # adds "test" to `WORKDIR`/relativeDir/
ADD test /absoluteDir/         # adds "test" to /absoluteDir/
COPY test relativeDir/         # adds "test" to `WORKDIR`/relativeDir/
COPY test /absoluteDir/        # adds "test" to /absoluteDir/
```


All new files and directories are created with a UID and GID of 0, unless the optional `--chown` flag specifies a given username, groupname, or UID/GID combination to request specific ownership of the content added. The format of the `--chown` flag allows for either username and groupname strings or direct integer UID and GID in any combination. Providing a username without groupname or a UID without GID will use the same numeric UID as the GID.

```
ADD --chown=55:mygroup files* /somedir/
ADD --chown=bin files* /somedir/
ADD --chown=1 files* /somedir/
ADD --chown=10:11 files* /somedir/
COPY --chown=55:mygroup files* /somedir/
COPY --chown=bin files* /somedir/
COPY --chown=1 files* /somedir/
COPY --chown=10:11 files* /somedir/
```


The first encountered `ADD` or `COPY` instruction will invalidate the cache for all following instructions from the Dockerfile if the contents of `<src>` have changed.


`ADD` and `COPY` obey the following rules:

- The `<src>` path must be inside the *context* of the build;   you cannot `ADD ../something /something` or `COPY ../something /something`, because the first step of a `docker build` is to send the context directory (and subdirectories) to the  docker daemon.
- If `<src>` is a directory, the entire contents of the directory are copied, including filesystem metadata.
- If `<src>` is any other kind of file, it is copied individually along with   its metadata. In this case, if `<dest>` ends with a trailing slash `/`, it   will be considered a directory and the contents of `<src>` will be written   at `<dest>/base(<src>)`.
- If multiple `<src>` resources are specified, either directly or due to the use of a wildcard, then `<dest>` must be a directory, and it must end with   a slash `/`.
- If `<dest>` does not end with a trailing slash, it will be considered a regular file and the contents of `<src>` will be written at `<dest>`.
- If `<dest>` doesn't exist, it is created along with all missing directories in its path.


Although `ADD` and `COPY` are functionally similar, generally speaking, `COPY` is preferred. That’s because it’s more transparent than `ADD`. `COPY` only supports the basic copying of local files into the container, while `ADD` has some features (like local-only tar extraction and remote URL support) that are not immediately obvious. If you have multiple `Dockerfile` steps that use different files from your context, `COPY` them individually, rather than all at once. This ensures that each step’s build cache is only invalidated (forcing the step to be re-run) if the specifically required files change.


For example:

```
COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
COPY . /tmp/
```

Results in fewer cache invalidations for the RUN step, than if you put the COPY . /tmp/ before it.

For other items (files, directories) that do not require `ADD`’s tar auto-extraction capability, you should always use `COPY`.


#### Volume

`VOLUME ["/data"]`

The `VOLUME` instruction should be used to expose any database storage area, configuration storage, or files/folders created by your docker container to the host machine.  The `VOLUME` instruction creates a mount point with the specified name and marks it as holding externally mounted volumes from native host or other containers. The value can be a JSON array, `VOLUME ["/var/log/"]`, or a plain string with multiple arguments, such as `VOLUME /var/log` or `VOLUME /var/log /var/db`.  Without the volume command, by default, all files created inside a container are stored on a writable container layer. This means that:

- The data doesn’t persist when that container is no longer running, and it can be difficult to get the data out of the container if another process needs it.
- A container’s writable layer is tightly coupled to the host machine where the container is running. You can’t easily move the data somewhere else.
- Writing into a container’s writable layer requires a storage driver to manage the filesystem. The storage driver provides a union filesystem, using the Linux kernel. This extra abstraction reduces performance as compared to using data volumes, which write directly to the host filesystem.

Volumes however, are stored in a part of the host filesystem which is managed by Docker (/var/lib/docker/volumes/ on Linux).  They are created and managed by Docker. You can create a volume explicitly using the `docker volume create` command, or Docker can create a volume during container or service creation. When you create a volume, it is stored within a directory on the Docker host. When you mount the volume into a container, this directory is what is mounted into the container.

A given volume can be mounted into multiple containers simultaneously. When no running container is using a volume, the volume is still available to Docker and is not removed automatically. You can remove unused volumes using `docker volume prune`.

Volumes are contrasted with bind mounts.  Bind mounts have limited functionality compared to volumes, but when you use a bind mount, a file or directory on the *host machine* is mounted into a container.  In other words, Volumes are a way to persist data created within the container, while bind mounts are a way for a container to interact with the host machine's filesystem.

One side effect of using bind mounts, for better or for worse, is that you can change the **host** filesystem via processes running in a **container**, including creating, modifying, or deleting important system files or directories. This is a powerful ability which can have security implications, including impacting non-Docker processes on the host system.

Volumes are the better choice when:

- You want to share data among multiple containers.
- You want to store your container’s data on a remote host or a cloud provider, rather than locally
- You need to back up, restore, or migrate data from one Docker host to another, volumes are a better choice.

Bind mounts are the better choice when:

- You want to share configuration files from the host machine to containers
- You want to share source code or build artifacts between a development environment on the Docker host and a container
- You need to access data from multiple parts of the host filesystem, but don't want to add the data to the container

![](images/types-of-mounts.png)

bind mounts are specified when you run the image for the first time, using the `--mount` flag.  You can also use the `-v` or `--volume` flag, but that is no longer the preferred method.  You can use `--mount` for both bind mounts and volumes. `--mount` consists of multiple key-value pairs, separated by commas and each consisting of a `<key>=<value>` tuple

- The `type` of the mount, which can be `bind`, `volume`, or `tmpfs`
- The source of the mount, may be specified as `source` or `src`. For named volumes, this is the name of the volume. For anonymous volumes, this field is omitted. For bind mounts, this is the path to the file or directory on the Docker daemon host.
- The destination of the moount, which may be specified as `destination`, `dst`, or `target`, takes as its value the path where the file or directory is mounted in the container. May .
- The `readonly` option, if present, causes the bind mount to be mounted into the container as read-only.
- The `volume-opt` option, which can be specified more than once, takes a key-value pair consisting of the option name and its value.

#### USER

`USER <user>[:<group>]`

OR

`USER <UID>[:<GID>]`

The `USER` instruction sets the user name (or UID) and optionally the user group (or GID) to use when running the image and for any `RUN`, `CMD` and `ENTRYPOINT` instructions that follow it in the `Dockerfile`.

If a service can run without privileges, use `USER` to change to a non-root user. Start by creating the user and group in the `Dockerfile` with something like `RUN groupadd -r postgres && useradd --no-log-init -r -g postgres postgres` or `RUN useradd -ms /bin/bash <username>`. Avoid installing or using sudo as it has unpredictable TTY and signal-forwarding behavior that can cause problems.

#### WORKDIR

`WORKDIR /path/to/workdir`

The `WORKDIR` instruction sets the working directory for any `RUN`, `CMD`, `ENTRYPOINT`, `COPY` and `ADD` instructions that follow it in the `Dockerfile`. If the `WORKDIR` doesn't exist, it will be created even if it's not used in any subsequent `Dockerfile` instruction.

The `WORKDIR` instruction can be used multiple times in a `Dockerfile`. If a relative path is provided, it will be relative to the path of the previous `WORKDIR` instruction. For example:

```
WORKDIR /a
WORKDIR b
WORKDIR c
RUN pwd
```

The output of the final `pwd` command in this `Dockerfile` would be
`/a/b/c`.

For clarity and reliability, you should always use absolute paths for your `WORKDIR`. Also, you should use `WORKDIR` instead of proliferating instructions like `RUN cd … && do-something`.  Be careful when using `WORKDIR` to create directories coupled with `USER`, as even though you set a user, if you create a directory with `WORKDIR` it will be owned by root.  Instead create the directory using `RUN`, which respects `USER`, and then enter the directory using `WORKDIR`.





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