# -*- coding: utf-8 -*-  
import requests
import pandas as pd
import numpy as np
from kashgari import utils
from flask import Flask, url_for, render_template, request, redirect, session
from keywordMapping import keyWordMapping

app = Flask(__name__)

model_path = './chat_model_serving/1'
url = "http://106.54.166.111:8501/v1/models/gru_bert:predict"

print('loading question mapping file ... ', end = '')
answer_mapping_file = 'answer.csv'
answer = pd.read_csv(answer_mapping_file,sep=',',encoding='utf-8')
answer = answer.groupby('question').agg('first')
print('complete!')

print('loading keyword mapping file', end = ' ... ')
keywordEngine = keyWordMapping()
keywordEngine.load_config_file('./data/keywords.csv')
km_question = '胸痛什么时候明显一些'
km_label = keywordEngine.judge(km_question)
print('test question = ', km_question, ', label = ',km_label)

def get_ai_reply(sentence,
                 model_path='../data/chat_model_serving/1',
                 url = "http://106.54.166.111:8501/v1/models/gru_bert:predict",
                 answer = '../data/answer.csv'):
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
    label_ai,prob_ai,reply_ai = get_ai_reply(question,model_path = model_path,url = url,answer = answer)
    print(label_ai,prob_ai,reply_ai)
    result = reply_ai
    result = result + '\n(label_ai：' + str(label) + ', probability：' + str(int(prob*100)) + '%)'
    
    label_keyword = keywordEngine.judge(question)
    result = result + '\n(label_keyword: ' + str(label_keyword)
    return result

@app.route('/wx', methods=['GET','POST'])
def wx():
    question=request.args.get('question')
    label_ai,prob_ai,reply_ai = get_ai_reply(question,model_path = model_path,url = url,answer = answer)
    print(label_ai,prob_ai,reply_ai)
    label_keyword = keywordEngine.judge(question)
    result_json = {'reply_ai':reply_ai,'label_ai':label_ai,'prob_ai':prob_ai,'label_keyword',label_keyword}
    return result_json
        
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.debug = True
    app.secret_key = "123"
    app.run(host='0.0.0.0',port=80)
    #app.run(host='0.0.0.0',ssl_context=('./ssl/nginx/1_www.pdctest.xyz_bundle.crt', './ssl/nginx/2_www.pdctest.xyz.key'))