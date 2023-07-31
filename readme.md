KRX stock prediction
========================
- used transformer, Prophet, LSTM model
- developed in linux 22.04
- used tensorflow 2.x   

### Tasks
--------------------------
***Junseo Min*** : model develop, data prescale, pipeline develop   
***hong yoon kyo*** : model evaluating, sales algorithm develop   
***Ann sung jin*** : economical knowledge shareing   

### Model Introduction
--------------------------
1. LSTM
    - LSTM is a type of recurrent neural network (RNN) suitable for handling sequential data. It is used to predict future stock prices by learning the sequential patterns in historical stock price data. LSTM is particularly useful for capturing long-term dependencies and effectively dealing with time lags and complex patterns in stock data.
2. Prophet   
    - Prophet is a time series forecasting tool developed by Facebook. It is specialized in handling time series data with strong seasonality and multiple seasonal patterns. When applied to stock prediction, Prophet models the overall trend of stock prices and predicts future price movements based on historical patterns.
3. Transformer
    - Transformer is an attention-based neural network architecture initially proposed for natural language processing tasks. However, it has been successfully applied to sequential data forecasting tasks as well. Transformer models utilize attention mechanisms to capture dependencies between different time steps, making it effective at modeling long-range interactions. For stock prediction, a Transformer-based model learns complex relationships between historical stock prices and trading volumes to forecast future stock movements.
  
