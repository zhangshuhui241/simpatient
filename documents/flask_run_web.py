# -*- coding: utf-8 -*-  
import requests
import pandas as pd
import numpy as np
from kashgari import utils
from flask import Flask, url_for, render_template, request, redirect, session

app = Flask(__name__)

model_path = './chat_model_serving/1'
url = "http://127.0.0.1:8501/v1/models/gru_bert:predict"

print('loading question loading file ... ', end = '')
answer_mapping_file = 'answer.csv'
answer = pd.read_csv(answer_mapping_file,sep=',',encoding='utf-8')
answer = answer.groupby('question').agg('first')
print('complete!')

def get_ai_reply(sentence,
                 model_path='./chat_model_serving/1',
                 url = "http://127.0.0.1:8501/v1/models/gru_bert:predict",
                 answer = './answer.csv'):
    sentence = list(sentence)
    processor = utils.load_processor(model_path = model_path)
    tensor = processor.process_x_dataset([sentence])
    tensor = [{"Input-Token:0": i.tolist(),"Input-Segment:0": np.zeros(i.shape).tolist()} for i in tensor]
    req = requests.post(url, json={"instances": tensor})
    preds = req.json()['predictions']
    labels = processor.reverse_numerize_label_sequences(np.array(preds).argmax(-1))
    reply = answer.loc[int(labels[0]),'answer']
    prob = np.array(preds).max()
    return labels[0],prob,reply

@app.route('/ai', methods=['GET','POST'])
def ai():
    question=request.args.get('question')
    label,prob,reply = get_ai_reply(question,model_path = model_path,url = url,answer = answer)
    print(label,prob,reply)
    result = reply + '  (question label：' + str(label) + ', probability：' + str(int(prob*100)) + '%)'
    return result

@app.route('/wx', methods=['GET','POST'])
def wx():
    question=request.args.get('question')
    label,prob,reply = get_ai_reply(question,model_path = model_path,url = url,answer = answer)
    print(label,prob,reply)
    result_json = {'reply':reply,'label':label,'prob':prob}
    return result_json
        
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.debug = True
    app.secret_key = "123"
    app.run(host='0.0.0.0',port=80)
    #app.run(host='0.0.0.0',ssl_context=('./ssl/nginx/1_www.pdctest.xyz_bundle.crt', './ssl/nginx/2_www.pdctest.xyz.key'))