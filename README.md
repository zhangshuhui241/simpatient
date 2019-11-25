

Prepare the tf model for sentence classification
1. Train the kashgari model
2. Turn the saved model to tensorflow serving format
3. Upload the model file to simpatient/chat_model/

Implement tensorflow serving:
1. Install docker application
2. docker pull tensorflow/serving
3. Start tensorflow serving 
	docker run -p 8501:8501 \
	  --mount type=bind,\
	   source=/tmp/tfserving/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_cpu,\
	target=/models/half_plus_two \
	-e MODEL_NAME=half_plus_two -t tensorflow/serving &

Setup the flask web server
1. upload flask_run_web.py #this is the main web app code
2. upload keywordMapping.py and keywords.csv #this is the for the old keyword mapping function
3. upload ./static/* and ./templates/* #this is for web pages



