gcloud compute ssh msnow@v1 --ssh-flag="-L 8898:localhost:8898"
gcloud compute ssh msnow@v1

wget https://raw.githubusercontent.com/MichoelSnow/data_science/master/containers/gpu-instance-setup-default-settings.sh
sh gpu-instance-setup-default-settings.sh


wget https://raw.githubusercontent.com/MichoelSnow/data_science/master/containers/gpu-instance-setup-default-settings-cuda-9-1.sh
sh gpu-instance-setup-default-settings-cuda-9-1.sh


sudo docker build -f Dockerfile_plus -t testing2  .

sudo docker run --runtime=nvidia -it -p 8898:8898 testing2

sudo docker run --runtime=nvidia --mount type=bind,source="$(pwd)"/data,target=/home/msnow/data -it -p 8898:8898 cnn

sudo docker run --runtime=nvidia --mount type=bind,source="$(pwd)"/data,target=/home/msnow/data -it -p 8898:8898 testing:v6

jupyter notebook --ip 0.0.0.0 --no-browser --port 8898 --notebook-dir=~

sudo docker run --runtime=nvidia --mount type=bind,source="$(pwd)"/data,target=/home/msnow/data -it -p 8899:8899 testing:v6

jupyter notebook --ip 0.0.0.0 --no-browser --port 8899 --notebook-dir=~

gcloud  compute scp dogscats_split.tar.gz msnow@v1:~/data/
tar xzf dogscats_split.tar.gz


apt-cache search nvidia | grep -P '^nvidia-[0-9]+\s'

sudo chown -R msnow ~/data/

sudo docker exec -it blissful_cori bash
sudo docker exec -u root -it relaxed_sammet bash
sudo docker commit blissful_cori updated_nb


gcloud compute instances stop v1

gcloud compute scp ~/.kaggle/kaggle.json msnow@gpu1-1:~/data/

cp data/kaggle.json .kaggle/

sudo apt-get update
sudo apt-get install -y p7zip-full
7za x test-jpg.tar.7z
7za x train-jpg.tar.7z
tar -xf test-jpg.tar
tar -xf train-jpgF.tar
unzip -q train_v2.csv.zip

# Planet dataset

kaggle competitions download -c planet-understanding-the-amazon-from-space -p \path\to\download\directory -f train-jpg.tar.7z
kaggle competitions download -c planet-understanding-the-amazon-from-space -p \path\to\download\directory -f test-jpg.tar.7z
kaggle competitions download -c planet-understanding-the-amazon-from-space -p \path\to\download\directory -f test-jpg-additional.tar.7z
kaggle competitions download -c planet-understanding-the-amazon-from-space -p \path\to\download\directory -f train_v2.csv.zip
kaggle competitions download -c planet-understanding-the-amazon-from-space -p \path\to\download\directory -f test_v2_file_mapping.csv.zip