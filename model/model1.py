#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#%%
data = pd.read_csv('./token_data.csv')
train_input = data['token']
train_target = data['labels']
print(train_input[:100])
#%%

train_input,test_input,train_target,test_target = train_test_split(train_input,train_target,test_size=.2,random_state=0)
#%%
train_input,val_input,train_target,val_target = train_test_split(train_input,train_target,test_size=.2,random_state=0)
#%%
print(train_input.shape)
print(test_input.shape)
print(val_input.shape)
# %%
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_input)
train_input_encoded = tokenizer.texts_to_sequences(train_input)
test_input_encoded = tokenizer.texts_to_sequences(test_input)
val_input_encoded = tokenizer.texts_to_sequences(val_input)
# tokenizer.fit_on_texts(train_input)
# tokenizer.fit_on_texts(train_input)
# %%
# print(train_input_encoded[:5])
word_to_index = tokenizer.word_index
vocab_size = len(word_to_index)+1
print(vocab_size)
# %%
print('본문의 최대 길이 :',max(len(sent) for sent in train_input[:1]))
print(train_input[:1])
print('test의 최대 길이 :',max(len(sent) for sent in test_input))
print('val의 최대 길이 :',max(len(sent) for sent in val_input))
print('본문의 평균 길이 :',sum(map(len, train_input))/len(train_input))
plt.hist([len(sent) for sent in train_input], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
# %%
max_len = 100
# %%
train_input_encoded = pad_sequences(train_input_encoded, maxlen=max_len, padding='post')
test_input_encoded = pad_sequences(test_input_encoded, maxlen=max_len,padding='post')
test_input_encoded = pad_sequences(test_input_encoded, maxlen=max_len,padding='post')
# %%
# print(train_input[1])
train_input_encoded[:1]