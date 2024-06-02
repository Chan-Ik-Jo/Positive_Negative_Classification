#%%
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pickle
import re
#%%
data = pd.read_csv('../finance_data.csv')
#결측치 제거
# %%
print(data[:5])
#%%# white space 데이터를 empty value로 변경
print(data.isnull().sum())
#%%
data = data.dropna()
# print("결측값 여부 : ",data.isnull().values.any())
# data.isnull()
# data.isnull().sum()# 결측치갯수 확인
print('kor_sentence 열의 유니크한 값 :',data['kor_sentence'].nunique())
duplicate = data[data.duplicated()]
duplicate
print(data.groupby('labels').size().reset_index(name='count'))
#%%
print(data.isnull().sum())
data['labels'].value_counts().plot(kind='bar')
#%%
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
8
from konlpy.tag import Mecab
from konlpy.tag import Okt

okt = Okt()
data['token'] = data['kor_sentence'].apply(okt.morphs)
#%%
# data['token'] = data['kor_sentence'].apply(mecab.morphs)
# print(data)
data.to_csv('token_data.csv')
train_input = data['token']
train_target = data['labels']
#%%
train_input[:5]
#%%
#train데이터 / test데이터
train_input,test_input,train_target,test_target = train_test_split(train_input,train_target,test_size=.2,random_state=0)
#%%
#train 데이터 / validation 데이터
train_input,val_input,train_target,val_target = train_test_split(train_input,train_target,test_size=.2,random_state=0)
#%%
print(train_input.shape)
print(test_input.shape)
# print(val_input.shape)
# print(train_input[:5])

# %%
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_input)
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
# val_input_encoded = tokenizer.texts_to_sequences(val_input)
train_input_encoded = tokenizer.texts_to_sequences(train_input)
test_input_encoded = tokenizer.texts_to_sequences(test_input)
# tokenizer.fit_on_texts(train_input)
# tokenizer.fit_on_texts(train_input)
# %%
# print(train_input_encoded[:1])
word_to_index = tokenizer.word_index
vocab_size = len(word_to_index)+1
print(vocab_size)
# %%
print('본문의 최대 길이 :',max(len(sent) for sent in train_input))
print('test의 최대 길이 :',max(len(sent) for sent in test_input))
print('본문의 평균 길이 :',sum(map(len, train_input))/len(train_input))
plt.hist([len(sent) for sent in train_input], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
# %%
max_len = 100

# %%
train_input_encoded = pad_sequences(train_input_encoded, maxlen=max_len, padding='pre')
test_input_encoded = pad_sequences(test_input_encoded, maxlen=max_len,padding='pre')
# %%
# print(train_input[1])
# train_input_encoded[:1]
# train_input_encoded.shape
train_target.shape
# %%

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import keras
import gc
_ = gc.collect()
tf.keras.backend.clear_session()
#%%
#%%

embedding_dim = 64
hidden_units = 64
#vocab_size = 97538
model = Sequential()
# model.add(Embedding(vocab_size,embedding_dim,input_length=100))
# model.add(LSTM(hidden_units))
# model.add(Dense(1,activation='sigmoid'))
model.add(keras.layers.Embedding(vocab_size,64))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Conv1D(32,3,padding="same",strides=1,activation='relu'))
model.add(keras.layers.MaxPool1D(2))
model.add(keras.layers.LSTM(100))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(1, activation='sigmoid'))
# model.compile(optimizer = 'Adam', loss='binary_crossentropy', metrics=['acc'])
# checkpoint_cb = keras.callbacks.ModelCheckpoint('best-lstm-model.h5',save_best_only=True)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

# %%

model.summary()
#%%
model.compile(optimizer = 'Adam', loss='binary_crossentropy', metrics=['acc'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-lstm-model.h5',save_best_only=True)
history = model.fit(train_input_encoded, train_target, epochs=15, callbacks=[es,mc], batch_size=32, validation_split=0.2)

# %%
plt.plot(history.history['loss'], 'y', label='train loss')
plt.plot(history.history['val_loss'], 'r', label='val loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
# %%
# validation = model.predict(val_input_encoded)

# %%

# %%
print(val_loss)
print(val_acc)
#%%
# input_str=pd.DataFrame({})
input_str = data['kor_sentence'][1:2]
print(input_str)
input_str = input_str.apply(lambda x : okt.morphs(x, stem=True))
input_str_encoded = tokenizer.texts_to_sequences(input_str)
input_str_encoded = pad_sequences(input_str_encoded, maxlen=max_len, padding='pre')

print(input_str_encoded)
#%%
pred = model.predict(val_input_encoded[4:5])
print(pred)
print(val_target[4:5])
# %%
import tensorflow as tf
model= tf.keras.models.load_model('best_model.h5')

#%%
from tensorflow.keras.utils import plot_model

plot_model(model, to_file='model_shapes.png', show_shapes=True)
#%%
# val_loss, val_acc = model.evaul
# %%
from konlpy.tag import Okt
import numpy as np

okt = Okt()

input_str = input('>>')
token_input = okt.morphs(input_str)
print(len(token_input))

input_dic = {}
input_dic['input'] = token_input
print(input_dic)
encoded = tokenizer.texts_to_sequences(input_dic['input'])
# print(encoded)
zzin_encoded = [item for item in encoded if item]
print(zzin_encoded)
#%%

zzin = [np.array(sum(zzin_encoded,[]))]
print(zzin)
# zzin = [zzin.flatten()]
#%%
print(zzin )
#%%
input_str_encoded = pad_sequences(zzin, maxlen=max_len, padding='pre')

# print(train_input_encoded[:1])
print(input_str_encoded)

#%%
zzinzzin = model.predict(input_str_encoded)
print(zzinzzin)
if zzinzzin>0.5:
    zzinzzin = 1
elif zzinzzin<0.5:
    zzinzzin =0
print(zzinzzin)
# print(train_input_encoded[:1])
# print(train_target[:1])
# model.fit(test_input_encoded,test_target,epochs=5, callbacks=[es,mc], batch_size=32)
#%%
# val_loss,val_acc = model.evaluate(test_input_encoded,test_target)
test_pre = model.predict(test_input_encoded)
prec = [0 if i < 0.5 else 1 for i in test_pre]
# %%
from sklearn.metrics import *
# print(prec)
# print(test_target)
cm = confusion_matrix(test_target, prec)
print(cm)
# %%
import seaborn as sns
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('predict')
plt.ylabel('answer')
plt.show()
report = classification_report(test_target, prec)

# %%
print(report)