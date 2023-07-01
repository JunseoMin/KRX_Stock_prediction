import pandas as pd
import numpy as np

# for Model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split,KFold
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler 
from sklearn import preprocessing

# using tree struct


class node_model:
    outputs=0
    ret_value=0
    
    def __init__(self,path="",model="") -> None:      
        self.data=pd.read_csv(path)
        self.model=model
        
        pass
    
    def prescaler(self):
        
        
        pass
    
    def study(self):
        pass
    
    