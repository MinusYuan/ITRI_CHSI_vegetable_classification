sudo docker rm $(sudo docker ps -aq) -f
sudo docker rmi $(sudo docker images -aq)
