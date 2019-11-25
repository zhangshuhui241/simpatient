

# 1. Prepare the tf model for sentence classification
## 1.1 Train the kashgari model
train on local ML server
## 1.2 Turn the saved model to tensorflow serving format
do it on local server
## 1.3 Upload the model file to simpatient/chat_model/
do it by ftp connection 120.31.132.238

# 2. Setup centos system
## 2.1 install python3.6
yum install zlib  zlib-devel #安装zlib模块，后期make install需要用到
mkdir -p ~/tmp
cd ~/tmp
wget https://www.python.org/ftp/python/3.6.6/Python-3.6.6.tgz
tar xzf Python-3.6.6.tgz
cd ~/tmp/Python-3.6.6
./configure --prefix=$HOME/python-3.6.6
make
make install

## 2.2 install docker


# 2. Implement tensorflow serving:
## 2.1 Install docker application
sudo apt install docker #for ubuntu
yum install docker #for centos
120.31.132.238

2.2 docker pull tensorflow/serving
2.3 Start tensorflow serving 
	docker run -p 8501:8501 \
	  --mount type=bind,\
	   source=/tmp/tfserving/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_cpu,\
	target=/models/half_plus_two \
	-e MODEL_NAME=half_plus_two -t tensorflow/serving &

# 3. Setup the flask web server
3.1 upload flask_run_web.py #this is the main web app code
3.2 upload keywordMapping.py and keywords.csv #this is the for the old keyword mapping function
3.3 upload ./static/* and ./templates/* #this is for web pages



