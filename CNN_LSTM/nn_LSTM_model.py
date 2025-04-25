import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def init_weights(m, init_type="kaiming"):
    if isinstance(m, nn.Linear):
        if init_type=="kaiming":
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif init_type=="xavier":
            nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)

class LSTMI(nn.Module):
    def __init__(self, input_channels, num_classes, max_length_series, num_lstm_layers=2, 
                 hidden_size=32, size_linear_lyr=32):
        """
        LSTM model for time series classification.
        Args:
            input_channels (int): Number of input channels.
            num_classes (int): Number of output classes.
            max_length_series (int): Maximum length of the time series.
            num_lstm_layers (int): Number of LSTM layers.
            hidden_size (int): Size of the hidden state in LSTM.
            size_linear_lyr (int): Size of the linear layer.
        """
        super(LSTMI, self).__init__()
        
        self.num_lstm_layers = num_lstm_layers
        self.hidden_size = hidden_size
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList()

        # for i in range(num_lstm_layers):
        #     input_size = input_channels if i == 0 else hidden_size
        #     self.lstm_layers.append(nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                                        #    batch_first=True, dropout=0.2))
        self.lstm = nn.LSTM(input_size=input_channels, hidden_size=hidden_size,
                            num_layers=num_lstm_layers, 
                            batch_first=True, dropout=0.2)

        
        # Calculate the size of the flattened output
        self.fc_input_size = hidden_size 
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, size_linear_lyr)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(size_linear_lyr, num_classes)
    
    def forward(self, x):
        # Input shape: (batch_size, time, channels)
        batch_size = x.size(0)

        h0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size).to(x.device)  # Hidden state
        c0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size).to(x.device)

        x, _ = self.lstm(x, (h0, c0))
        
        # Fully connected layers
        x = F.relu(self.fc1(x[:,-1,:]))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x