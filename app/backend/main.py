from fastapi import FastAPI,Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle
from konlpy.tag import Okt
import numpy as np


max_len=100
model = tf.keras.models.load_model('../../model/best_model.h5')
with open('../../model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
    
app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:5500",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/proc")
async def get_str(client_str:str):
    okt = Okt()
    token_input = okt.morphs(client_str)
    # print(token_input)
    input_dic = {}
    input_dic['input'] = token_input
    # print(input_dic)
    encoded = tokenizer.texts_to_sequences(input_dic['input'])
    # print(encoded)
    zzin_encoded = [item for item in encoded if item]
    # print(zzin_encoded)
    zzin = [np.array(sum(zzin_encoded,[]))]
    # print(zzin)
    input_str_encoded = pad_sequences(zzin, maxlen=max_len, padding='pre')
    # print(train_input_encoded[:1])
    # print(input_str_encoded)
    zzinzzin = model.predict(input_str_encoded)
    print(zzinzzin*100)
    result = {"Response" : int(zzinzzin[0][0]*100)}
    return result