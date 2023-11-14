import pandas as pd

def test1():
    data = pd.read_csv('./fake-news/train.csv')
    data1 = pd.read_csv('./fake-news/test.csv')
    print(data)
    print(data1)
    pass
test1()