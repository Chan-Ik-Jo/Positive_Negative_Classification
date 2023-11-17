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
def get_data():
    # os.chdir('./model')
    train_data = pd.read_csv('../fake-news/train.csv')
    test_data   = pd.read_csv('../fake-news/test.csv'  )
    test_label = pd.read_csv('../fake-news/submit.csv')
    
    train_input= np.array(list(data_clean(train_data))+list(data_clean(test_data)))
    train_target = np.array(list(get_label(train_data))+list(get_label(test_label)))
    
    return train_input,train_target
# %%
a,b = get_data()
print(a)
print(b)
# %%
np.save('./save_data/train',a)
np.save('./save_data/train_label',b)


# %%
