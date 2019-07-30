#!/usr/bin/env bash

# Built from k80-cnn snapshot
gcloud compute --project "symbolic-base-138223" disks create "k80" --size "130" --zone "us-east1-c" --source-snapshot "k80-cnn" --type "pd-ssd"

gcloud beta compute --project=symbolic-base-138223 instances create k80 --zone=us-east1-c --machine-type=n1-highmem-4 --subnet=default --network-tier=PREMIUM --no-restart-on-failure --maintenance-policy=TERMINATE --preemptible --service-account=375126418499-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append --accelerator=type=nvidia-tesla-k80,count=1 --tags=http-server,https-server --disk=name=k80,device-name=k80,mode=rw,boot=yes,auto-delete=yes


