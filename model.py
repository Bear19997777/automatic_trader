import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sqlite3 as sq 
import psycopg2
import dotenv
import os  
# 讀取股票數據
dotenv.load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("psql_host"),
    database=os.getenv("psql_database"),
    user=os.getenv("psql_user"),
    password=os.getenv("psql_pswd")
)
# df = pd.read_csv("stock_data.csv")
df = pd.read_sql_query("SELECT * FROM price", conn)
df  = df[df["stock_id"]=='0015']
# df = pd.read_sql_table("price",conn)
# 將數據進行歸一化處理
scaler = MinMaxScaler()
data = scaler.fit_transform(df[['收盤價']].values)

# 定義訓練和測試數據集
train_data = data[:]
test_data = data[:]

# 定義LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x.view(len(x), 1, -1))
        y_pred = self.linear(lstm_out.view(len(x), -1))
        return y_pred[-1]

model = LSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 訓練模型
epochs = 50
for epoch in range(epochs):
    train_loss = 0.0
    for i in range(len(train_data)-30):
        x = torch.tensor(train_data[i:i+30], dtype=torch.float32)
        y_true = torch.tensor(train_data[i+30], dtype=torch.float32)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f'Epoch {epoch+1}, Train Loss: {train_loss}')

# 預測股票價格
y_pred = []
for i in range(len(test_data)-30):
    x = torch.tensor(test_data[i:i+30], dtype=torch.float32)
    with torch.no_grad():
        y_pred.append(model(x).item())

# 將預測結果轉換回原始數據的格式
y_pred = scaler.inverse_transform(np.array(y_pred).reshape(-1, 1))

# 繪製預測結果
# plt.figure(figsize=(12, 6))
# plt.plot(df['date'][800+30:], test_data[30:], label='Actual Price')
# plt.plot(df['date'][800+30:], y_pred, label='Predicted Price')
# plt.xlabel('Date')
# plt.ylabel('Stock Price')
# plt.legend()
# plt.show()

