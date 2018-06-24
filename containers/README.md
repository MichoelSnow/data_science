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

For this project the container is meant to use CUDA on an nvidia GPU on ubuntu (can also be centOS, I just chose ubuntu), so my `FROM` command is

```
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
```

Since we are going to need the nvidia deep learning SDK to support CUDA we want to add the nvidia machine learning repos to our sources list

```
RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
```

Then install the packages we need

```
RUN apt-get update && apt-get install -y --allow-downgrades --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libnccl2=2.0.5-3+cuda9.0 \
         libnccl-dev=2.0.5-3+cuda9.0 \
         python-qt4 \
         libjpeg-dev \
         zip \
         unzip \
         sudo \
         wget \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*
```






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

 To get out of a running container without stoppping it use <kbd>CTRL</kbd>+<kbd>p</kbd>, <kbd>CTRL</kbd>+<kbd>q</kbd> to turn interactive mode into daemon mode.  To re-enter the container use the appropriate `docker container exec` command



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



# Google Cloud

## Setting up your account for GPU computing

### Signing up for a free account

If you ahve never used Google Cloud Platform before then you are probably eligibile for a [free trial](https://console.cloud.google.com/freetrial?authuser=2&page=0) which gives you access to all the services and $300 in credit to spend over the next 12 months.   You will still need to enter your credit card info, but you won't be charged until you go over the $300 credit limit.

### Setting your default region

Each of your VM instances will live in a specific region and zone.  Normally you would just set these values to the location closes to you, but each region/zone combination has access to different resources.   Go the documentation page for [Regions and Zones](https://cloud.google.com/compute/docs/regions-zones/regions-zones?hl=en_US&_ga=2.77309013.-1507351947.1529583865) to find out the resources available in each of the zones, for the GPUs you need to go to the [GPUs on Compute Engine](https://cloud.google.com/compute/docs/gpus/) documentation page.  At the time of writing, this was the breakdown of resources available by zone in region *us-east1*, according to the google documentation, which is the closest region to me with GPUs:



| Zone                  | Features   | GPUs|
| --------------------- |-------------| ------ |
| us-east1-b        | 2.3 GHz Intel Xeon E5 v3 (Haswell) platform (default) <br>   2.2 GHz Intel Xeon E5 v4 (Broadwell) platform <br>   2.0 GHz Intel Xeon (Skylake) platform <br>  Up to 96 vCPU machine types when using the Skylake platform <br>  [Local SSDs](https://cloud.google.com/compute/docs/disks/local-ssd) <br> Memory-optimized machine types with up to 160 vCPUs and 3.75 TB of system memory| NVIDIA Tesla P100
| us-east1-c        | 2.3 GHz Intel Xeon E5 v3 (Haswell) platform (default) <br>   2.2 GHz Intel Xeon E5 v4 (Broadwell) platform <br>   2.0 GHz Intel Xeon (Skylake) platform <br>  Up to 96 vCPU machine types when using the Skylake platform <br>  [Local SSDs](https://cloud.google.com/compute/docs/disks/local-ssd) <br> [GPUs](https://cloud.google.com/compute/docs/gpus) <br> [Sole-tenant nodes](https://cloud.google.com/compute/docs/nodes/) | NVIDIA Tesla P100  <br> NVIDIA Tesla K80
| us-east1-d         |  2.3 GHz Intel Xeon E5 v3 (Haswell) platform (default) <br>   2.2 GHz Intel Xeon E5 v4 (Broadwell) platform <br>   2.0 GHz Intel Xeon (Skylake) platform <br>  Up to 96 vCPU machine types when using the Skylake platform <br>  [Local SSDs](https://cloud.google.com/compute/docs/disks/local-ssd) <br> [GPUs](https://cloud.google.com/compute/docs/gpus) <br> [Sole-tenant nodes](https://cloud.google.com/compute/docs/nodes/)| NVIDIA Tesla K80

If your looked at the above table you will probably notice that it is not self-consistent, i.e., *us-east1-b* doesn't have GPUs listed in its features yet it supposedly has access to *NVIDIA Teslap P100*.  Don't worry about this as it gets even more self-inconsistent when we actually get to selecting the GPUs.

You can set your default region and zone in the [settings page](https://console.cloud.google.com/compute/settings) of the Google Compute Engine section. You can always change your default zone and region, it is just good practice to set by default, so that all your resources will match up.


### Requesting access to a GPU

In order to attach a GPU to your VM you first need to request a quota increase (as you start with a quote of 0 GPUs).  You first need to go to the [quotas page](https://console.cloud.google.com/iam-admin/quotas?_ga=2.220448216.-1507351947.1529583865) in the admin section of the console.  The first time you go there you will rpobably see a notification near the top telling you that *You can't request an increase until you upgrade your free trial account.* and giving you the option to upgrade your account.  If you upgrade your account you still get to keep the $300 credit, the only caveat is that now if you go over the free credit you will be automatically charged after you use your credits or they expire.  As you'll see below when we set up an instance, that you can get an GPU instance running at somehwere between $0.20 to $0.50 per hour.  So there is no need to worry about running out of the credit too quickly, and as long as you keep an eye on the Billing section of the console to make sure you still have credit, you'll be fine.


Once you have upgraded your account you can now select a GPU quote to increase.  This is where the region and zone matter, as your GPU quota will be specific to a region.  So if your VM instance is in *us-east1* then your GPU quota will need to be in that region as well.  On the top right of the quotas page you can limit your selection of resources to whatever region you want.  I'll be working with the *us-east1* region.  After you select your region you will still be left with a lot of choices.  Most of the rows will have 0/<some number> in the used column, this means that you have unuqued quota of those resources.  If you go down to the GPUs, their used column should show -/0, indicating that you have no alloted quota of GPUs.  In the us-east1 region these were the GPUs available at the time of writing.


- NVIDIA K80 GPUs
- NVIDIA P100 GPUs
- NVIDIA V100 GPUs
- Preemptible NVIDIA K80 GPUs
- Preemptible NVIDIA P100 GPUs
- Preemptible NVIDIA V100 GPUs

You can see that in addition to the regular GPU's there are also preemptible GPUs.  Preemptible GPUs are GPUs added to a preemptible VM instance.  You can read all about [Preemptible VM Instances}(https://cloud.google.com/compute/docs/instances/preemptible) but the gist is that a preemptible instance is usually around half the cost of a non-preemtible instance, but Compute Engine might terminate (preempt) these instances if it requires access to those resources for other tasks.   Additionally, Compute Engine always terminates preemptible instances after they run for 24 hours.  Your data is still persisted on the VM when it is preempted.  So if you are using the VM instance for something like a jupyter notebook, and you make sure to save your data along the way, then a preemptible instance might just be for you.

Also, you might notice that NVIDIA V100 GPUs showed up on the list of possible GPUs, even though they are not listed in the documentation as being available in this region.  This is why I take all the documentation with a grain of salt, as the resources are constantly changing.

Getting back to the quotas, the general advice I have found so far is to not be too aggressive in the quotas you ask for.  Start with only one or two GPUs and then only ask for a new quota limit of 1 (unless you initially need more).  After selecting the boxes of your GPUs there is a button at the top of the page to *Edit Quotas*. Here they will ask you for your new quota limits for each GPU selected and then a rquest description.  I tend to put something like "research" as the description.  The google cloud compute team will then get back to you, usually in under a day, to, hopefully, let you know that they've increased your quotas as per your request.


## Creating a VM instance

You can create an instance directly in the Cloud Compute [VM instance](https://console.cloud.google.com/compute/instances) subsection, but I would actually advise you to instead create a template in the [instances templates](https://console.cloud.google.com/compute/instanceTemplates/) section.  This way you can easily spin up new instances without having to set every setting each time.

Here are the settings which you will need to modify based on your own needs.

### Machine type

Select customize to go into the advanced view.  My general base settings here are 4 vCPU, 26 GB of memory and for the GPU I tend to choose 1 NVIDIA Tesla K80.

### Container

Even though we are going to be using docker containers, I tend to skip this option as it is intended more for exclusively running a container on your instance.  We will still be able to run our docker on the instance even if we don't directly deploy it.  If you do want to deploy a container from docker hub the address for the container image is as follows:

```
registry.hub.docker.com/<username>/<repo>:<tag>
```

### Boot disk

Choose whatever os you are comfortable with.  It is probably easiest to choose the os that matches the os of your docker container.  In my case the docker container is built on top of the Nvidia ubuntu 16.04 container, so that is the os I am going to choose for my VM.

At the bottom, you can choose the type of disk (standard vs SSD) and it's size.  I tend to go with 50 GB on an SSD.

### Firewall

You want to check both boxes to allow for HTTP and HTTPS traffic

### Management, disks, networking SSH keys

#### Management

This is where you can set whether or not you want a preemptible instance.  By default an instance is not preemptible, but you can set preemptibility to *on* to change that.

#### SSH Keys

you can add your SSH keys here or you can specify them for the whole project in the SSH key tab of the [Metadata](https://console.cloud.google.com/compute/metadata/sshKeys) section in the Google compute engine.

If you do nothave an existing private na dmatching publc SSH key file, you will need to generate a new SSH key.  THe google cloud documentation of [creating a new SSH key](https://cloud.google.com/compute/docs/instances/adding-removing-ssh-keys#createsshkeys) is quite good, so I'll just cover the basics to save the reader the effort of clicking on the link (plus there is never any guarantee on how long that page will exist)

##### Linux and Mac

Open a terminal and use the `ssh keygen` command to generate a new key.  Google cloud likes to have a username associated with the key so you will need to use the `-c` flag.

```
ssh-keygen -t rsa -f ~/.ssh/[KEY_FILENAME] -C [USERNAME]
```

`[KEY_FILENAME]` is the name that you want to use for your SSH key files. For example, a filename of *my-ssh-key* generates a private key file named *my-ssh-key* and a public key file named *my-ssh-key.pub*.

You then want to restrict access to your private key so that only you can read it and nobody can write to it.

```
chmod 400 ~/.ssh/[KEY_FILENAME]
```

You will be asked for a passphrase to generate the private key.  See this [serverfault response](https://serverfault.com/questions/142959/is-it-okay-to-use-a-ssh-key-with-an-empty-passphrase) for a discussion regarding the need for SSH key passphrases.  There is also the ssh-agen which can make it more convenient to use a ssh key with a passphrase.


#### Windows

Windows does not natively have a tool for generating SSH keys, but you can use a 3rd party tool to generate one, such as git or puttygen.  I'm going to be walking through the puttygen version, but here is a [link](https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/) to the gitbash instruction and here is a post on [configuring ssh to work in powershel](https://dillieodigital.wordpress.com/2015/10/20/how-to-git-and-ssh-in-powershell/).

[Puttygen](http://www.chiark.greenend.org.uk/~sgtatham/putty/download.html)  is a tool for generating SSH keys.  If you have putty installed then you already have puttygen, otherwise you need to [download it](http://www.chiark.greenend.org.uk/~sgtatham/putty/download.html).  Find and run the `puttygen.exe` file, which will open a new window to configure your key settings.

Click Generate and follow the on-screen instructions to generate a new key. For most cases, the default parameters are fine, but you must generate keys with at least 2048 bits. When you are done generating the key, the tool displays your public key value.  In the Key comment section, replace the existing text with the username of the user for whom you will apply the key. Optionally, you can enter a Key passphrase to protect your key. Click Save private key to write your private key to a file with a .ppk extension. Click Save public key to write your public key to a file for later use. Keep the PuTTYgen window open for now.  Note that if you created an SSH key with PuTTYgen, the default public SSH key file will not be formatted correctly if it is opened outside of PuTTYgen, so to view it you will need to open putty gen and load your key.


For all OS's, once you have generated your SSH key you want to copy the public key into a new SSH key on your metadata page.  When you are done hit save at the bottom of the page.

### Starting a VM instance

Once you have all your template settings set as you would lik ethem you can save your template.  You template should now be listed in the [instances templates](https://console.cloud.google.com/compute/instanceTemplates/) section.   If you click on the 3 dots to the right of your template, one of the options should be to *create VM*, you can also create a vm from a template by clicking on the template and then clicking on the Create VM option at the top of the temaplte summary page.

## Connecting to your instance

The OS of your container will determine how you can connect to it.  This guide will go over connecting to linux based instances using the `gcloud` command line tool.  For other methods of connecting to Linux instances as well as for methods to connect to Windoes instances, see the google cloud documentation on [Connecting to Instances](https://cloud.google.com/compute/docs/instances/connecting-to-instance).  Before you connect using `gcloud` you have to install the sdk and set up `gcloud` compute.

### Installing gcloud

There is a different installation process for each OS, as explained in the documentation for the [Google Cloud SDK](https://cloud.google.com/sdk/docs/).  Follow the steps for whichever OS you are running on your machine.

For the most common gcloud commands and their usage see [gcloud_commands.md](gcloud_commands.md)

#### Set up gcloud compute

Here is the link to the [documentation](https://cloud.google.com/compute/docs/gcloud-compute/#auth)


Google Compute Engine uses OAuth2 to authenticate and authorize access. Before you can use `gcloud compute`, you must first authorize the Cloud SDK on your behalf to access your project and acquire an auth token.  If you are using the `gcloud` command-line tool for the first time, `gcloud` automatically uses the default configuration. For most cases, you only need the `default` configuration.

Run `gcloud init` to start the authentication process. Hit enter when prompted.  The command prints a URL and tries to open a browser window to request access to your project. If a browser window can be opened, you will see the following output:

```
Welcome! This command will take you through the configuration of gcloud.

Your current configuration has been set to: [default]


To continue, you must login. Would you like to login (Y/n)?  y


Your browser has been opened to visit:


https://accounts.google.com/o/oauth2/auth?scope=https%3A%2F%2Fwww.googleapis.co%2
Fauth%2Fappengine.admin+https%3A%2F%2...
```



If the Cloud SDK detects that a browser can not be opened (e.g., you are working on a remote machine) you will see the output below. Or, if you are working on a local machine and your browser doesn't automatically load the URL, then retry the `gcloud init` command with the `--console-only` flag:

```
Go to the following link in your browser:

https://accounts.google.com/o/oauth2/auth?scope=https%3A%2F%2Fwww.googleapis.co%2
Fauth%2Fappengine.admin+https%3A%2F%2...


Enter verification code:
```
Copy the authentication URL and paste it into a browser. Then paste the verification code back into the terminal.

After setting up your credentials, `gcloud` prompts for a default project for this configuration. Select a project ID from the list.

After you set this property, all of your `gcloud compute` commands use the default project ID unless you override it with the `--project flag` or set the `CLOUDSDK_CORE_PROJECT` environment variable. If you do not set a default project or environment variable, you must include a `--project` flag in each `gcloud compute` command that you run.

When you run gcloud for the first time, it also sets a default zone and default region for you, based on the default zone and region keys in your project metadata.   If it is not set, `gcloud` will ask you to provide it with each request.


### gcloud compute ssh

```
gcloud compute ssh [USER@]INSTANCE [--ssh-flag=SSH_FLAG]
```

`gcloud compute ssh` is the main command you will be using to connect to your instance.  `gcloud compute ssh` is a thin wrapper around the `ssh(1)` command that takes care of authentication and the translation of the instance name into an IP address. I am going to be going over my most common uses, but for the full docuemtnation see [gcloud compute ssh](https://cloud.google.com/sdk/gcloud/reference/compute/ssh).


### Arguments

- `USER`
  - username with which to SSH. If left blank will use the $USER from the environment.
- `INSTANCE`
  - VM instance to SSH into.  Once you have initialized gcloud and linked to a project, you can ssh with just the instance name, and don't need to know the actual IP address.
- `--ssh-flag=SSH_FLAG`
  - Additional flags to be passed to ssh(1). It is recommended that flags be passed using an assignment operator and quotes

#### Example

```
gcloud compute ssh msnow@instance-1 --ssh-flag="-L 8898:localhost:8898"
```

This connects to *instance1* as user *msnow* and the SSH flag connects port *8898* on the remote machine to my local port *8898*.  I most commonly use this setup when I want to run Jupyter notebooks on the VM instance.



### Confirm docker version

run `docker -v` to check your version of docker.  If docker is not isnallted or less than version 1.12 you will need to install docker on your VM instance.  Follow the instruction in the previous [Install and Start Docker](#install-and-start-docker) section.


## Install Nvidia CUDA drivers

Follow the step in the CUDA toolkit documentation for the installation appropriate to your OS.  Here I am following the [Nvidia CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#abstract)


### Pre-installation Actions

- Verify You Have a CUDA-Capable GPU
  - `lspci | grep -i nvidia` should return something like `00:04.0 3D controller: NVIDIA Corporation GK210GL [Tesla K80] (rev a1)`
- Verify You Have a Supported Version of Linux
  - `uname -m && cat /etc/*release` should return a version that matches one of the ones supported
- Verify the System Has gcc Installed
  - `gcc --version` should return something like `gcc (Ubuntu 5.4.0-6ubuntu1~16.04.9) 5.4.0 20160609`
  - If this returns an error like, `The program 'gcc' is currently not installed. To run 'gcc' please ask your administrator to install the package 'gcc'`, then you can install gcc on ubuntu with the following command `sudo apt-get install build-essential`
- Verify the System has the Correct Kernel Headers and Development Packages Installed
  - While the Runfile installation performs no package validation, the RPM and Deb installations of the driver will make an attempt to install the kernel header and development packages if no version of these packages is currently installed. However, it will install the latest version of these packages, which may or may not match the version of the kernel your system is using. Therefore, it is best to manually ensure the correct version of the kernel headers and development packages are installed prior to installing the CUDA Drivers, as well as whenever you change the kernel version.
  - The version of the kernel your system is running can be found by running the following command `uname -r`
  - On ubuntu the kernel headers and development packages for the currently running kernel can be installed with `sudo apt-get install linux-headers-$(uname -r)`
- Choose an Installation Method
  - The CUDA Toolkit can be installed using either of two different installation mechanisms: distribution-specific packages (RPM and Deb packages), or a distribution-independent package (runfile packages). The distribution-independent package has the advantage of working across a wider set of Linux distributions, but does not update the distribution's native package management system. The distribution-specific packages interface with the distribution's native package management system. It is recommended to use the distribution-specific packages, where possible.


### Package Manager Installation

- Download the CUDA toolkit
  - go to the [CUDA toolkit archive](https://developer.nvidia.com/cuda-toolkit-archive) and find the toolkit that matches the CUDA version of your container
  - Select your target platform and for *Installer Type* choose deb (local)
  - Copy the links for the installer and then download them to your VM instance with something like `wget`
  - At the bottom of the page, below the buttons to download the installers, there should also be the documentation for that specific version, soemthing like [Installation Guide for Linux](http://developer.download.nvidia.com/compute/cuda/9.0/Prod/docs/sidebar/CUDA_Installation_Guide_Linux.pdf)
- Install repository metadata
  - `sudo dpkg -i cuda-repo-<distro>_<version>_<architecture>.deb`
  - For my docker and os this becomes
    - `sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb`
  - You can go to Nvidia's [CUDA repo](https://developer.download.nvidia.com/compute/cuda/repos/) to find the one that matches your settings
- Installing the CUDA public GPG key
  - The previous command should have suggested a command to run to install the public CUDA GPG key, something like `sudo apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub`
  - If not you can use the general form `sudo apt-key add /var/cuda-repo-<version>/7fa2af80.pub` where version is your CUDA toolkit version, e.g., 9-0-local
  - If it works it should return `OK`
- Update the Apt repository cache and install CUDA
  - `sudo apt-get update`
  - `sudo apt-get install cuda`
- Reboot the system to load the NVIDIA drivers.

### Post-installation Actions

- The PATH variable needs to include /usr/local/cuda-<version>/bin
  -  `export PATH=/usr/local/cuda-<varsion>/bin${PATH:+:${PATH}}`
    - `export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}`
- Verify the integrity of the installation (optional)
  - Install Writable Samples
    - `cuda-install-samples-9.0.sh ~`
  - Verify the Driver Version
    - `cat /proc/driver/nvidia/version`
  - Compiling the Examples
    - `cd  ~/NVIDIA_CUDA-9.0_Samples`
    - `make`
  - Running the Binaries
    - `bin/x86_64/linux/release/deviceQuery`
      - The important outcomes are that a device was found , that the device matches the one on your system, and that the test passed.
      - On systems where SELinux is enabled, you might need to temporarily disable this
security feature to run deviceQuery. To do this, type `setenforce 0`
    - 'bin/x86_64/linux/release/bandwidthTest'
      - This test  ensures that the system and the CUDA-capable device are able to communicate correctly
      -  The important point is that you obtain measurements, and that the second-to-last line confirms that all necessary tests passed


## NVIDIA Container Runtime for Docker

Before you can utilize the NVIDIA drivers in your container you need to install the NVIDIA container runtime.  The runtime allows driver agnostic CUDA images and provides a Docker command line wrapper that mounts the user mode components of the driver and the GPU device files into the container at launch.  See these links for more information on [docker container runtime theory](https://devblogs.nvidia.com/gpu-containers-runtime/) and the process for [installing the runtime](https://github.com/NVIDIA/nvidia-docker).

To make things easier I have included a shell script to install the runtime for you, it is the [nvidia-docker-ubuntu.sh](nvidia-docker-ubuntu.sh) file in this directory

## Building your container on the VM instance

[Containers on Compute Engine](https://cloud.google.com/compute/docs/containers/)




[gcloud compute instances list](https://cloud.google.com/sdk/gcloud/reference/compute/instances/list)
[pushing a docker image to google container repository](https://stackoverflow.com/questions/20429284/how-do-i-run-docker-on-google-compute-engine)
[Setting a root password for a Docker image created with USER](https://www.kevinhooke.com/2015/11/08/setting-a-root-password-for-a-docker-image-created-with-user/)
[How to install latest nvidia drivers in Linux](http://www.linuxandubuntu.com/home/how-to-install-latest-nvidia-drivers-in-linux)





```
sudo docker build -f Dockerfile -t fastai_dl .
docker pull msnow/nn_benchmark:v1
docker run -it -p 8898:8898 msnow/nn_benchmark:v1
jupyter notebook --ip 0.0.0.0 --no-browser --port 8898
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

