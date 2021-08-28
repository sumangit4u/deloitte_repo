#!bin/bash 
sudo docker pull sumandocker4u/deloitte:latest
sudo docker container run --publish 5049:5000 --detach sumandocker4u/deloitte:latest
echo "Docker running on localhost port 5049"
sudo curl "http:localhost:5049/"
