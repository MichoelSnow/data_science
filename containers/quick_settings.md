These are the steps to quickly setup the nvidia runtime environment and drivers using the default settings.  This assumes that you have the gcloud sdk installed on your local machine.  If not see the [Google Cloud SDK ](https://cloud.google.com/sdk/docs/) docs.

# Local Terminal

## Authorize account

Confirm that your desired account is authorized

```
gcloud auth list
```

If the desired account does not show up on the list, you can add it with

```
gcloud auth login [ACCOUNT] [Options]
```

Where `[ACCOUNT]` will typically be your email address, and if you don't want it to try and launch a browser use the `--no-launch-browser` flag.

## Connect to instance

Check if the instance you want to connect to exists and is running

```
gcloud compute instances list
```

If it does not exist, you can create it through the [gcloud cli](https://cloud.google.com/sdk/gcloud/reference/compute/instances/create) but I think it is much easier to create an instance using the [google cloud console](https://console.cloud.google.com/compute/instances)

If your desired instance exists but is not running, you can start it with

'''
gcloud compute instances start [INSTANCES]
'''

You can then connect to your desired instance using

```
gcloud compute ssh [INSTANCE]
```

If you need to forward the port for something like a jupyter notebook use

```
gcloud compute ssh [INSTANCE] --ssh-flag="-L 8898:localhost:8898"
```


# VM Instance

## Runing the script
Once connected to the VM instance you want to get the quick settings script

```
wget https://raw.githubusercontent.com/MichoelSnow/data_science/master/containers/gpu-instance-setup-default-settings.sh
```

Which you can then run

```
sh gpu-instance-setup-default-settings.sh
```

If everything completes without error, you will need to restart the instance, which you can do with `sudo reboot`.

## Checking the docker container

Once you have reconnected to the restarted instance, you can check tha the docker container is set up correctly by first entering it

```
sudo docker run --runtime=nvidia -it -p 8898:8898 testing
```

Then starting `python`, and checking that you get similar responses to the commands below

```python
In [1]: import torch

In [2]: torch.cuda.current_device()
Out[2]: 0

In [3]: torch.cuda.device(0)
Out[3]: <torch.cuda.device at 0x7efce0b03be0>

In [4]: torch.cuda.device_count()
Out[4]: 1

In [5]: torch.cuda.get_device_name(0)
Out[5]: 'Tesla K80'
```

If you want to check that your jupyter notebook is forwarding correctly you can run

```
jupyter notebook --ip 0.0.0.0 --no-browser --port 8898
```