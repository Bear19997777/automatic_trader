import torch
import torch.nn as nn
import torch.optim as optim

train_x, train_y, test_x, one, label, df = data[0], data[1], data[2], data[3], data[4], data[5]

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=20, hidden_size=32, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(p=0.2)
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=32, num_layers=1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = self.flatten(out[:, -1, :])
        out = nn.functional.relu(self.fc1(out))
        out = nn.functional.relu(self.fc2(out))
        out = nn.functional.softmax(self.fc3(out), dim=1)
        return out

model = LSTMModel()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()

train_x = torch.tensor(train_x).float()
train_y = torch.tensor(train_y).float()

for epoch in range(40):
    model.train()
    optimizer.zero_grad()
    output = model(train_x)
    loss = criterion(output, train_y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}: Loss = {loss:.4f}")
    
torch.save(model.state_dict(), f'../my_h5_model/{stock_number}_my_model.pt')
