#%%
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import contractions
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import os
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
#%%
def row_merge(data): # 열병합
    data = data.fillna('')
    data["total"] = data['title'] + " " + data["author"]  
    return data

def data_sub(data):#특수문자 및 대소문자 가공
    ps = PorterStemmer()
    corpus = []
    for i in range(len(data)):
        m = re.sub("[^a-zA-Z]", " ", data["total"][i])
        m = m.lower()
        m = m.split()
        m = [ps.stem(word) for word in m if not word in stopwords.words('english')]
        clean_text = " ".join(m)
        corpus.append(clean_text)
    return corpus

def onehot(data, VOCAB_SIZE = 5000):
  return [one_hot(words, VOCAB_SIZE) for words in data]

def padding(onehot_text) :
    
    return np.array(pad_sequences(onehot_text, padding="pre", maxlen = 25))

def get_label(data):
    return np.array(data["label"])

def data_clean(origin_data):
    result = row_merge(origin_data)
    result = data_sub(result)
    result = onehot(result)
    result = padding(result)
    return result
#%%

#%%
def get_data():
    while(True):
        Title = input("제목 >> ")
        Author = input("작성자 >> ")
        Content = input("내용 >> ")
        break
    data = [[Title,Author,Content]]
    df = pd.DataFrame(data,columns=['title','author','txt'])
    print(df)
    input_txt= np.array(list(data_clean(df)))
    return input_txt