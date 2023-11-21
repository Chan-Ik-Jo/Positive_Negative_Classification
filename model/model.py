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
from tensorflow.python.keras.callbacks import EarlyStopping

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
        pass
    def data_split(self):
        train_input,test_input,train_target,test_target = train_test_split(self.train_input,self.train_target,test_size=0.2)
        train_input,val_input,train_target,val_target = train_test_split(train_input,train_target,test_size=0.2)
        return train_input,train_target,test_input,test_target,val_input,val_target
    
    def train(self):
        pass
    
    def pre(self,input_data):
        pass
    
    def model_frame(self):
        self.cnn_lstm.summary()
        
def main():
    train_input= np.load('./model/save_data/train.npy')
    train_label= np.load('./model/save_data/train_label.npy')
    model = Model(train_input,train_label)
    model.model_frame()
main()