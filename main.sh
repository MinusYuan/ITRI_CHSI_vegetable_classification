#!/bin/bash
echo "Running the services"
sudo docker compose up -d
echo "unzip the saved containers"
gunzip -c vegetable_counting_itri.tar.gz  | sudo docker load
echo "run the container"
sudo docker run -d --rm --gpus all -p 12345:12345 vegetable_counting_itri