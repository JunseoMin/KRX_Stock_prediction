import pandas as pd
import numpy as np

# for Model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split,KFold
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler 
from sklearn import preprocessing
from keras.layers import LSTM


# using tree struct

#RNN이니까 n번째 돌때마다 데이터를 노드중 한 노드의 학습 데이터를 많이 수정 - 이 부분 얼만큼 해야할지 정하기

class node_model:
    outputs=0
    ret_value=0
    
    def __init__(self,path="",model="") -> None:      
        self.data=pd.read_csv(path)
        self.model=model
        
        
        
        pass
    
    def prescaler(self):

        
        self.traindata=
        self.testdata=
        pass
    
    def study(self):
        model=Sequential()
        model.Add(LSTM(,activation='relu',input_shape=()))
        
        
        
    
        pass
    
