#%%
import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers as L
from tensorflow.keras.layers import  Embedding
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, GRU, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
#%%
class Model():
    def __init__(self,train_input,train_target):
        self.train_input,self.train_target=train_input,train_target
        self.cnn_lstm = Sequential()
        self.cnn_lstm.add(keras.layers.Embedding(30000,30))
        self.cnn_lstm.add(keras.layers.Dropout(0.3))
        self.cnn_lstm.add(keras.layers.Conv1D(32,3,padding="same",strides=1,activation='relu'))
        self.cnn_lstm.add(keras.layers.MaxPool1D(2))
        self.cnn_lstm.add(keras.layers.LSTM(100))
        self.cnn_lstm.add(keras.layers.Dropout(0.3))
        self.cnn_lstm.add(keras.layers.Dense(1, activation='sigmoid'))
        self.cnn_lstm.compile(optimizer = 'Adam', loss='binary_crossentropy', metrics=['acc'])
        self.checkpoint_cb = keras.callbacks.ModelCheckpoint('best-lstm-model.h5',save_best_only=True)
        pass
    def data_split(self):
        train_input,test_input,train_target,test_target = train_test_split(self.train_input,self.train_target,test_size=0.2)
        train_input,val_input,train_target,val_target = train_test_split(train_input,train_target,test_size=0.2)
        return train_input,train_target,test_input,test_target,val_input,val_target
    
    def train(self):
        train_input,train_target,test_input,test_target,val_input,val_target = self.data_split()
        history= self.cnn_lstm.fit(train_input, train_target, epochs=100,
                    validation_data=(val_input, val_target),callbacks=[self.checkpoint_cb])
        return history
    def pre(self,input_data):
        result = self.cnn_lstm.predict(input_data)
        return result
    
    def model_frame(self):
        self.cnn_lstm.summary()
#%%
train_input= np.load('./save_data/train.npy')
train_label= np.load('./save_data/train_label.npy')
model = Model(train_input,train_label)
model.model_frame()
history = model.train()
#%%
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], 'y', label='train loss')
plt.plot(history.history['val_loss'], 'r', label='val loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
train_input,train_target,test_input,test_target,val_input,val_target = model.data_split()
#%%
from input_data import get_data

user_txt = get_data()
# model.cnn_lstm.summary()
# model.cnn_lstm.evaluate(val_input,val_target)
result=model.pre(user_txt)
print(result)