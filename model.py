import VDN
import torch
import torch.nn as nn
import preprocessing as ps 
import dotenv
import torch.optim as optim
import numpy as np 
import pandas as pd 
# 讀取股票數據
dotenv.load_dotenv()
test_stock = 2303
ps = ps.datascience() 
data = ps.DII_analysis(f"{test_stock}")
device = torch.device("mps")
# MinMaxScaler
# train & test 

class LSTMModel(nn.Module):
# define LSTM model
    def __init__(self, input_size=8, hidden_size=32, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, num_layers=1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 3)
        self.flatten = nn.Flatten()

        # self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        
        out, _ = self.lstm1(x)
        out = nn.functional.relu(self.fc1(out))
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = nn.functional.relu(self.fc2(out))
        out = self.flatten(out[-1:, :])
        out = nn.functional.softmax(self.fc3(out), dim=1)
        return out


model = LSTMModel()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = nn.MSELoss()


train_x = torch.tensor(data["train_x"],dtype=torch.float32)
train_y = torch.tensor(data["train_y"],dtype=torch.float32)
epochs = 15
print("data training.....")
for epoch in range(epochs):
    for x,y in zip(train_x,train_y):
        model.train()
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()

    print(f"Epoch {epoch+1}: Loss = {loss:.6f}")
    
# torch.save(model.state_dict(), f'../my_h5_model/{stock_number}_my_model.pt')


# predict model 
y_pred = []
model = model.eval()
for x in data["test_x"]:
    x = torch.tensor(x).float()
    with torch.no_grad():
        output = model(x)
        maxindex = output.argmax(dim=1, keepdim=True)
        pred = [0,0,0]
        pred[maxindex] = 1 
        y_pred.append(pred)

onehot_inverse = data["onehot"].inverse_transform(y_pred)
label_inverse = data["label"].inverse_transform(onehot_inverse)
inverse_result = pd.DataFrame(label_inverse, columns=["pred_y"])
df = data["test_df"].reset_index().join(inverse_result)
vdn_dict = VDN.VDN(df, "pred_y")


# print(y_pred)

# inverse
# plot predictions
# plt.figure(figsize=(12, 6))
# plt.plot(df['date'][800+30:], test_data[30:], label='Actual Price')
# plt.plot(df['date'][800+30:], y_pred, label='Predicted Price')
# plt.xlabel('Date')
# plt.ylabel('Stock Price')
# plt.legend()
# plt.show()

