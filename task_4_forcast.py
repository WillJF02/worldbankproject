import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


#MAPEloss function
class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        #ensure no division by zero
        epsilon = 1e-7
        return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100

#LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

#CNN-LSTM model
class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, lstm_hidden_dim, output_dim, kernel_size, n_layers):
        super(CNNLSTMModel, self).__init__()
        self.cnn = nn.Sequential(
        nn.Conv1d(in_channels=input_dim, out_channels=cnn_out_channels, kernel_size=kernel_size),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2)
        )
        self.lstm = nn.LSTM(cnn_out_channels, lstm_hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
    
#transformer positional encoding model
class PositionalEncoding(nn.Module):
    def __init__(self, num_features):
        super(PositionalEncoding, self).__init__()
        timespan = 10
        self.pe = np.zeros(shape = (timespan,num_features))
        
        Y = np.arange(0,timespan,1)
        X = np.arange(0,num_features,2)
        i,pos = np.meshgrid(X,Y)

        sin_val = np.sin(pos / (10000 ** (i / num_features)))
        cos_val = np.cos(pos / (10000 ** ((i + 1) / num_features)))

        self.pe[:,::2] = sin_val
        self.pe[:,1::2] = cos_val
        self.pe = torch.tensor(self.pe, dtype=torch.float32)
                
    def forward(self, x):
        return x + self.pe.unsqueeze(0)

#transformer model
class TransformerModel(nn.Module):
    def __init__(self, num_features, num_heads, num_layers, output_size):
        super(TransformerModel, self).__init__()
        
        self.positional_encoding = PositionalEncoding(num_features)
       
        # Transformer Encoder Layer
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=num_features, nhead=num_heads),
            num_layers=num_layers
        )
        
        # Output layer
        self.output_layer = nn.Linear(num_features, output_size)
        
    def forward(self, x):
        x = self.positional_encoding(x)  # Apply positional encoding here
        x = self.transformer_encoder(x)  # Pass through transformer encoder
        x = x[:, -1, :]  # Use the output of the last time step
        output = self.output_layer(x)  # Pass through output layer
        return output