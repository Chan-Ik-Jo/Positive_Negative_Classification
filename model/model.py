import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
from tensorflow.keras.layers import  Embedding
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, GRU, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping


class Model():
    def __init__(self):
        self.cnn_lstm = Sequential()
        self.cnn_lstm.add(keras.layers.Embedding())
        self.cnn_lstm.add(keras.layers.LSTM())
        self.cnn_lstm.add(keras.layers.Conv1D())
        self.cnn_lstm.add(keras.layers.MaxPool1D())
        self.cnn_lstm.add(keras.layers.Dropout())
        self.cnn_lstm.add(keras.layers.Dense(1, activation='sigmoid'))
        self.cnn_lstm.compile(optimizer = 'Adam', loss='binary_crossentropy', metrics=['acc'])
        
    