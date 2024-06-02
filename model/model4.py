#%%
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#%%
data = pd.read_csv('../finance_data.csv')
data = data.dropna() #결측치 제거


# %%
print(data)
#%%
# drop_data = data.drop(,axis=0)
# print(drop_data)

# print("결측값 여부 : ",data.isnull().values.any())
# data.isnull()
data.isnull().sum()# 결측치갯수 확인
print('kor_sentence 열의 유니크한 값 :',data['kor_sentence'].nunique())
duplicate = data[data.duplicated()]
# duplicate

print(data.groupby('labels').size().reset_index(name='count'))
#%%
data['labels'].value_counts().plot(kind='bar')
#%%
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# from konlpy.tag import Mecab
from konlpy.tag import Okt

# mecab = Mecab('C:\\mecab\\mecab-ko-dic')
okt = Okt()
data['token'] = data['kor_sentence'].apply(okt.morphs)
#%%
# print(data)
data.to_csv('token_data.csv')
train_input = data['token']
train_target = data['labels']
#%%
#train데이터 / test데이터
train_input,test_input,train_target,test_target = train_test_split(train_input,train_target,test_size=.2, random_state=0)
#%%
#train 데이터 / validation 데이터
train_input,val_input,train_target,val_target = train_test_split(train_input,train_target,test_size=.2, random_state=0)
#%%
# print(train_input.shape)
# print(test_input.shape)
# print(val_input.shape)
print(train_input[:5])

# %%
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_input)
train_input_encoded = tokenizer.texts_to_sequences(train_input)
test_input_encoded = tokenizer.texts_to_sequences(test_input)
val_input_encoded = tokenizer.texts_to_sequences(val_input)
# tokenizer.fit_on_texts(train_input)
# tokenizer.fit_on_texts(train_input)
# %%
print(train_input_encoded[:1])
word_to_index = tokenizer.word_index
vocab_size = len(word_to_index)+1
print(vocab_size)
# %%
print('본문의 최대 길이 :',max(len(sent) for sent in train_input))
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
val_input_encoded = pad_sequences(val_input_encoded, maxlen=max_len,padding='post')
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
model.compile(optimizer = 'Adam', loss='binary_crossentropy', metrics=['acc'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-lstm-model2.h5',save_best_only=True)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

# %%
model.summary()
#%%
model.compile(optimizer = 'Adam', loss='binary_crossentropy', metrics=['acc'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-lstm-model2.h5',save_best_only=True)
history = model.fit(train_input_encoded, train_target, epochs=5, callbacks=[checkpoint_cb], batch_size=32, validation_split=0.2)

# %%
plt.plot(history.history['loss'], 'y', label='train loss')
plt.plot(history.history['val_loss'], 'r', label='val loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
# %%
validation = model.predict(val_input_encoded)

# %%
val_loss,val_acc = model.evaluate(val_input_encoded,val_target)
# %%
print(val_loss)
print(val_acc)
#%%
# input_str=pd.DataFrame({})
input_str = train_input[:1]
print(input_str)
input_str_encoded = tokenizer.texts_to_sequences(input_str)
input_str_encoded = pad_sequences(input_str_encoded, maxlen=max_len, padding='post')
print(input_str_encoded)
#%%

pred = model.predict(input_str_encoded)
# %%
print(pred)
print(train_target[:1])
# %%
input_str={}
input_str['input'] = okt.morphs("'친윤 핵심' 장제원, 오전 총선 불출마 공식 선언")
# %%
print(input_str)
input_str_encoded = tokenizer.texts_to_sequences(input_str['input'])
input_str_encoded = pad_sequences(input_str_encoded, maxlen=max_len, padding='post')
print(input_str_encoded)

#%%

import tensorflow as tf

model = tf.keras.models.load_model('best-lstm-model2.h5')
#%%
from konlpy.tag import Okt
import numpy as np

okt = Okt()

input_str = input('>>')
token_input = okt.morphs(input_str)
print(token_input)

input_dic = {}
input_dic['input'] = token_input

print(input_dic)
encoded = tokenizer.texts_to_sequences(input_dic['input'])
print(encoded)
zzin_encoded = [item for item in encoded if item]
print(zzin_encoded)
zzin = np.array(zzin_encoded)
zzin = [zzin.flatten()]

print(zzin)
input_str_encoded = pad_sequences(zzin, maxlen=max_len, padding='post')

# print(train_input_encoded[:1])
print(input_str_encoded)

zzinzzin = model.predict(input_str_encoded)
print(zzinzzin)
# %%
