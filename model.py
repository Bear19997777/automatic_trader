import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import preprocessing as ps 
import sqlite3 as sq 
import dotenv
import os  
import connectorx as cx 
# 讀取股票數據
dotenv.load_dotenv()

ps = ps.datascience() 
data = ps.DII_analysis()
# MinMaxScaler
# train & test 

class LSTM(nn.Module):
# define LSTM model
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size*2)
        self.linear1 = nn.Linear(hidden_size*2, hidden_size*4)
        self.dropout1 = nn.Dropout(0.2)        
        self.lstm2 = nn.LSTM(hidden_size*4,hidden_size*4)
        self.linear2 = nn.Linear(hidden_size*4, output_size)

        # self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x) #.view(len(x), 1, -1)
        linear_out = self.linear1(lstm_out)
        lstm_out,_ = self.lstm2(linear_out)
        y_pred = self.linear2(lstm_out)
        
        
        
        return y_pred[-1]

model = LSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train model 
epochs = 30
for epoch in range(epochs):
    train_loss = 0.0
    for i in range(len(train_data)-7):
        x = torch.tensor(train_data[i:i+7], dtype=torch.float32)
        y_true = torch.tensor(train_data[i+7], dtype=torch.float32)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f'Epoch {epoch+1}, Train Loss: {train_loss}')

# predict model 
y_pred = []
for i in range(len(test_data)-7):
    x = torch.tensor(test_data[i:i+7], dtype=torch.float32)
    with torch.no_grad():
        y_pred.append(model(x).item())

# inverse
y_pred = scaler.inverse_transform(np.array(y_pred).reshape(-1, 1))
for x,y in zip(y_pred,df['收盤價'][int(len(data)*0.5):]):
    print(f"pred_price:{x},real_price:{y}")
# plot predictions
# plt.figure(figsize=(12, 6))
# plt.plot(df['date'][800+30:], test_data[30:], label='Actual Price')
# plt.plot(df['date'][800+30:], y_pred, label='Predicted Price')
# plt.xlabel('Date')
# plt.ylabel('Stock Price')
# plt.legend()
# plt.show()

