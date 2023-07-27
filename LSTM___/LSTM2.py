import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime,timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense ,LSTM, Dropout,Bidirectional,TimeDistributed
from sklearn.model_selection import train_test_split, KFold
from tensorflow import expand_dims
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from os.path import join
from tensorflow import constant
from time import sleep
from prophet import Prophet
import tensorflow as tf


class LSTM2():
  def __init__(self,data,idx):
    self.data=data
    self.idx=idx
    self.time_steps=60

  def create_sequences(self, data):
    num_samples, num_features = data.shape
    sequences = []
    for i in range(num_samples - self.time_steps +1):
      sequences.append(data[i:i + self.time_steps, :])
    return np.array(sequences)

  def inverse_sequences(self,data_sequences):
    num_samples, num_steps, num_features = data_sequences.shape
    data = np.zeros((num_samples + self.time_steps - 1, num_features))
    for i in range(num_samples):
      data[i:i + self.time_steps, :] += data_sequences[i, :, :]
    data /= self.time_steps
    return data[:num_samples]
    #frquency domain inverse laplace transform

  def slicing_data(self):
    self.x_scaler = MinMaxScaler()
    self.y_scaler = MinMaxScaler()
    # Scaling features
    x_data = self.data[self.data.columns[:]]
    x_data = self.x_scaler.fit_transform(x_data)
    # Scaling target variable
    y_data = self.data[self.data.columns[:]]
    y_data = self.y_scaler.fit_transform(y_data)
    # validation set
    x_train_p, x_test, y_train_p, y_test = train_test_split(x_data, y_data, test_size=0.2, shuffle=False)
    x_train, x_val, y_train, y_val = train_test_split(x_train_p, y_train_p, test_size=0.25, shuffle=False)
    __t,x_final,__t,__t=train_test_split(x_data, y_data, test_size=60/len(y_data), shuffle=False)

    self.num_features = x_data.shape[1]
    self.num_samples=x_data.shape[0]

    self.x_train = self.create_sequences(x_train)
    self.x_test = self.create_sequences(x_test)
    self.x_val = self.create_sequences(x_val)
    self.x_final = self.create_sequences(x_final)

    self.y_train=self.create_sequences(y_train)
    self.y_test=self.create_sequences(y_test)
    self.y_val=self.create_sequences(y_val)

  def model_struct(self):
    self.model = Sequential()
    # layers
    self.model.add(Bidirectional(LSTM(1024, return_sequences=True, input_shape=(self.time_steps, self.num_features))))
    self.model.add(Dense(256, activation='relu'))
    self.model.add(Dense(128, activation='relu'))
    self.model.add(Dense(64, activation='relu'))
    self.model.add(Dense(32, activation='relu'))
    self.model.add(Dense(16, activation='relu'))
    # output layer (many-to-many with TimeDistributed)
    self.model.add(TimeDistributed(Dense(len(self.data.columns), activation='relu')))

  def train_model(self):
    model_save_path = '/content/drive/MyDrive/2023_1st_vac/KRX_modelings/best_model/LSTM2/'
    self.filename = join(model_save_path, 'checkpoint_0726_{}.ckpt'.format(self.idx))
    checkpoint = ModelCheckpoint(self.filename, save_weights_only=True, save_best_only=True, monitor='val_loss', verbose=0)
    earlystopping = EarlyStopping(monitor='val_loss', patience=100)
    self.model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
    self.history = self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=128, validation_data=(self.x_val, self.y_val), shuffle=False, callbacks=[checkpoint, earlystopping], verbose=0)


  def get_gap_by_test(self):
    self.model.load_weights(self.filename)
    pred = self.model.predict(self.x_test,verbose=0)
    rescaled_pred = self.y_scaler.inverse_transform(self.inverse_sequences(pred))
    rescaled_real = self.y_scaler.inverse_transform(self.inverse_sequences(self.y_test))

    gaps = []

    for p, r in zip(rescaled_pred, rescaled_real):
        gap = np.abs(p - r)
        gaps.append(gap)

    avg_gap = np.mean(gaps)

    return avg_gap



  def return_val(self):
    self.slicing_data()
    self.model_struct()
    self.train_model()
    self.model.load_weights(self.filename)

    next_input_data = np.copy(self.x_final)
    for day in range(1, 16):
        pred_day = self.model.predict(np.expand_dims(next_input_data[-1], axis=0),verbose=0)
        next_input_data = np.concatenate((next_input_data, pred_day), axis=0)

    next_15_days_data = next_input_data[-15:]
    next_15_days_data=self.y_scaler.inverse_transform(self.inverse_sequences(next_15_days_data))

    columns=self.data.columns.tolist()
    ret_df = pd.DataFrame(next_15_days_data, columns=columns)

    prices=[]
    for end_price in ret_df.iloc[:,-1]:
      prices.append(end_price)

    gap=self.get_gap_by_test()

    return prices,gap