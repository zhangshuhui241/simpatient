
# 1. Setup centos system

## 1.1 install python3.6 and packages
sudo yum -y install epel-release
sudo yum -y install python36
pip3 install kashgari
pip3 install tensorflow

## 1.2 install git
sudo yum install git

## 1.3 install docker
sudo yum install docker

## 1.4 install tensorflow serving
service docker start
docker pull tensorflow/serving


# 2. Prepare the tf model for sentence classification

## 2.1 Train the kashgari model
prepare/update answer.csv and samples.csv file, put them in ./ai_code
run model training code, for 10 epochs

## 2.2 Turn the saved model to tensorflow serving format
convert kashgari model to tf serving model

## 2.3 Upload the model file to simpatient/ai_code/tf_model_serving/
do it by ftp connection 120.31.132.238


# 3. Start the server

## 3.1 Start tensorflow serving 
docker run -p 8501:8501 \
  --mount type=bind,\
   source=/tmp/tfserving/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_cpu,\
target=/models/half_plus_two \
-e MODEL_NAME=half_plus_two -t tensorflow/serving &
	
## 3.2 Setup the flask web server




