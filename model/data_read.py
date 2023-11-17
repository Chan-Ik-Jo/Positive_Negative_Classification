import pandas as pd
import numpy as np
import re
import os

def test1():
    os.chdir('./model')
    # data = pd.read_csv('d:\\Python\\Winter_project\\fake-news\\train.csv') #절대 경로 입력 해야함
    data = pd.read_csv('../fake-news/train.csv') #절대 경로 입력 해야함
    # data = pd.read_csv('../fake-news/test.csv') #절대 경로 입력 해야함
    # print(data[0:1][:])
    # re.sub("\n"," ",data)
    # data = np.array(data)
    # print(data.shape)
    print(data)
    pass
test1()