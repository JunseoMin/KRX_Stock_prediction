import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from os.path import join
import tensorflow as tf


class Transformer():
    def __init__(self, data, idx):
        self.data = data
        self.idx = idx
        self.time_steps = 60
        
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


    def transformer_model(self):
        # Input layer
        inputs = Input(shape=(self.time_steps, self.num_features))
        
        # Transformer Encoder
        x = inputs
        for _ in range(4):  # Transformer Encoder Block의 반복 횟수 (하이퍼파라미터)
            x = self.transformer_encoder_block(x)
        
        # Output layer
        x = Dense(self.num_features, activation='relu')(x)
        
        model = Model(inputs, x, name="transformer_model")
        return model

    def transformer_encoder_block(self, inputs):
        # Multi-Head Self-Attention
        attention = tf.keras.layers.MultiHeadAttention(
            key_dim=self.num_features // 8, num_heads=8)(inputs, inputs)
        attention = Dropout(0.1)(attention)
        x = LayerNormalization(epsilon=1e-6)(inputs + attention)
        
        # Feed Forward Neural Network
        x = Dense(4 * self.num_features, activation="relu")(x)
        x = Dropout(0.1)(x)
        x = Dense(self.num_features, activation="relu")(x)
        x = LayerNormalization(epsilon=1e-6)(x + inputs)
        return x

    @tf.function(reduce_retracing=True)
    def train_model(self):
        model_save_path = '/content/drive/MyDrive/2023_1st_vac/KRX_modelings/best_model/Transformer2/'
        self.filename = join(model_save_path, 'checkpoint_0726_{}.ckpt'.format(self.idx))
        checkpoint = ModelCheckpoint(self.filename, save_weights_only=True, save_best_only=True, monitor='val_loss', verbose=0)
        earlystopping = EarlyStopping(monitor='val_loss', patience=100)
        model = self.transformer_model()
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))  # 학습률 (하이퍼파라미터)
        self.history = model.fit(self.x_train, self.y_train, epochs=1, batch_size=128, validation_data=(self.x_val, self.y_val), shuffle=False, callbacks=[checkpoint, earlystopping], verbose=0)

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
        model = self.transformer_model()  
        self.train_model()  
        model.load_weights(self.filename)  

        next_input_data = np.copy(self.x_final)
        for day in range(1, 16):
            pred_day = model.predict(np.expand_dims(next_input_data[-1], axis=0), verbose=0)
            next_input_data = np.concatenate((next_input_data, pred_day), axis=0)

        next_15_days_data = next_input_data[-15:]
        next_15_days_data = self.y_scaler.inverse_transform(self.inverse_sequences(next_15_days_data))

        columns = self.data.columns.tolist()
        ret_df = pd.DataFrame(next_15_days_data, columns=columns)

        prices = []
        for end_price in ret_df.iloc[:, -1]:
            prices.append(end_price)

        gap = self.get_gap_by_test()

        return prices, gap
'''
하이퍼파라미터 추천:
-->  논문 이상치  instanced
Transformer Encoder Block의 반복 횟수: 4
Multi-Head Self-Attention에서 key_dim: 입력 특성 수의 1/8 크기로 추천 (예: num_features=100 이라면 key_dim=12)
학습률 (learning_rate): 0.001 (Adam optimizer를 사용하는 경우 일반적으로 사용되는 값)
Dropout 비율: 0.1 (적절한 정도의 regularization을 위한 값)
Transformer Encoder Block의 뉴런 수 (Dense layer의 뉴런 수): 입력 특성 수의 몇 배로 할지 지정 (예: 입력 특성 수가 100이면 4 * num_features=400 정도로 지정)
'''
