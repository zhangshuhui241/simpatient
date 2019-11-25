#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import sys

def get_len(s):
    return (len(s.split(u' '))+len(s.split(u'#'))-2)

class keywordMapping():
    def __init__(self):
        self.df_input=None
    def load_config_file(self,fname):
        self.df_input=pd.read_csv(fname,encoding='gb18030')
        self.df_input[u'length']=list(map(lambda x:get_len(x), self.df_input[u'关键词']))
        self.df_input=self.df_input.sort_values(by=u'length',ascending=False)
        self.df_input=self.df_input.reset_index(drop=True)
    def judge(self,s):
        label=-1
        if self.df_input is None:
            print(u'pleas load label file!')
        else:
            old_count=0
            for index, r in self.df_input.iterrows():
                count=0
                for k in r[u'关键词'].split(u'#'):
                    rst=True
                    if k.strip()!=u'':
                        sub_rst=False
                        tmp_=k.strip().split(u' ')
                        for t in tmp_:
                            if t.strip()!=u"":
                                if s.find(t)>=0:
                                    sub_rst=True
                                    count+=1
                        if sub_rst==False:
                            rst=False
                            break
                if rst==True and count>old_count:
                    old_count=count
                    label=r[u'应答句序号']
                    #print(old_count,label)
                    #break

        return label#,old_count

def main():
    #sample code
    km=keywordMapping()
    km.load_config_file(r'../data/keywords.csv')
    question = '胸痛什么时候明显一些'
    label = km.judge(question)
    print(question, label)

main()