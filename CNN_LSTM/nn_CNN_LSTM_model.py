import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m, init_type="kaiming"):
    if isinstance(m, nn.Linear):
        if init_type=="kaiming":
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif init_type=="xavier":
            nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)



class CNN_LSTMI(nn.Module):
    def __init__(self, input_channels, num_classes,
                 max_length_series, num_conv_layers=2, 
                 initial_channels=8, num_lstm_layers=2, 
                 lstm_hidden_size=32, size_linear_lyr=32):
        super(CNN_LSTMI, self).__init__()
        
        self.num_conv_layers = num_conv_layers
        self.initial_channels = initial_channels
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        
        
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.lstm_layers = nn.ModuleList()

        in_channels = input_channels
        for i in range(num_conv_layers):
            self.conv_layers.append(nn.Conv1d(in_channels, initial_channels, kernel_size=3, padding=1))
            self.pool_layers.append(nn.MaxPool1d(kernel_size=2))
            
            # Update in_channels and out_channels for the next layer
            in_channels = initial_channels
            initial_channels *= 2  # Double the number of channels for the next layer
        initial_channels = int(initial_channels/2)

        # LSTM layers
        self.lstm = nn.LSTM(input_size=initial_channels, hidden_size=lstm_hidden_size,
                            num_layers=num_lstm_layers, 
                            batch_first=True, dropout=0.2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_hidden_size, size_linear_lyr)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(size_linear_lyr, num_classes)
    
    def forward(self, x):
        # Input shape: (batchsamples, time, channels)
        x = x.permute(0, 2, 1)  # (batch_size, input_channels, timesteps)
        
        # Apply convolutional and pooling layers
        for i in range(self.num_conv_layers):
            x = F.relu(self.conv_layers[i](x))
            x = self.pool_layers[i](x)

        # Input shape: (batch_size,channels, time): 
        x = x.permute(0, 2, 1)  # (batch_size, timesteps, channels)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_size).to(x.device)  # Hidden state
        c0 = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_size).to(x.device) # Cell state
        x, _ = self.lstm(x, (h0, c0))

        # Fully connected layers
        x = F.relu(self.fc1(x[:, -1, :]))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    


class CNNwithBatchNormLSTM(nn.Module):
    def __init__(self, input_channels, num_classes, 
                 num_conv_layers=2, size_linear_lyr=32, 
                 initial_channels=8, num_lstm_layers=2, 
                 lstm_hidden_size=32):
        super(CNNwithBatchNormLSTM, self).__init__()
        
        self.num_conv_layers = num_conv_layers
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()  # Batch Normalization layers
        self.pool_layers = nn.ModuleList()
        
        # Dynamically create convolutional and pooling layers
        in_channels = input_channels
        
        for i in range(num_conv_layers):
            self.conv_layers.append(nn.Conv1d(in_channels, initial_channels, kernel_size=3, padding=1))
            self.bn_layers.append(nn.BatchNorm1d(initial_channels))
            self.pool_layers.append(nn.MaxPool1d(kernel_size=2))
            
            # Update in_channels and out_channels for the next layer
            in_channels = initial_channels
            initial_channels *= 2  # Double the number of channels for the next layer
        initial_channels = int(initial_channels/2)
        # LSTM layers
        self.lstm = nn.LSTM(input_size=initial_channels, hidden_size=lstm_hidden_size,
                            num_layers=num_lstm_layers, 
                            batch_first=True, dropout=0.2)

        # Fully connected layers
        self.fc1 = nn.Linear(lstm_hidden_size, size_linear_lyr)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(size_linear_lyr, num_classes)
    
    def forward(self, x):
        # Input shape: (batchsamples, time, channels)
        x = x.permute(0, 2, 1)  # (batch_size, input_channels, timesteps)
        
        # Apply convolutional and pooling layers
        for i in range(self.num_conv_layers):
            x = self.conv_layers[i](x)
            x = self.bn_layers[i](x)    # Batch Normalization
            x = F.relu(x) 
            x = self.pool_layers[i](x)
        
        # Input shape: (batch_size,channels, time): 
        x = x.permute(0, 2, 1)  # (batch_size, timesteps, channels)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_size).to(x.device)  # Hidden state
        c0 = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_size).to(x.device)  # Cell state
        x, _ = self.lstm(x, (h0, c0))

        # Fully connected layers
        x = F.relu(self.fc1(x[:, -1, :]))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x



class CNNwithBatchNormLSTMParrallel(nn.Module):
    def __init__(self, input_channels, num_classes,
                 num_conv_layers=2, size_linear_lyr=32, 
                 initial_channels=8, num_lstm_layers=2, 
                 lstm_hidden_size=32):
        """
        CNN with Batch Normalization and LSTM model for time series classification.
        Args:
            input_channels (int): Number of input channels.
            num_classes (int): Number of output classes.
            num_conv_layers (int): Number of convolutional layers.
            size_linear_lyr (int): Size of the linear layer.
            initial_channels (int): Initial number of channels for the first convolutional layer.
            num_lstm_layers (int): Number of LSTM layers.
            lstm_hidden_size (int): Size of the hidden state in LSTM.
        """
        super(CNNwithBatchNormLSTMParrallel, self).__init__()
        
        self.num_conv_layers = num_conv_layers
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()  # Batch Normalization layers
        self.pool_layers = nn.ModuleList()
        
        # Dynamically create convolutional and pooling layers
        in_channels = input_channels
        
        for i in range(num_conv_layers):
            self.conv_layers.append(nn.Conv1d(in_channels, initial_channels, kernel_size=3, padding=1))
            self.bn_layers.append(nn.BatchNorm1d(initial_channels))
            self.pool_layers.append(nn.MaxPool1d(kernel_size=2))
            
            # Update in_channels and out_channels for the next layer
            in_channels = initial_channels
            initial_channels *= 2  # Double the number of channels for the next layer
        
        # Flatten layer
        self.flatten = nn.Flatten()
        self.fc_cnn = nn.LazyLinear(out_features=size_linear_lyr)

        # LSTM layers
        self.lstm = nn.LSTM(input_size=input_channels, hidden_size=lstm_hidden_size,
                            num_layers=num_lstm_layers, 
                            batch_first=True, dropout=0.2)
        self.fc_lstm = nn.Linear(lstm_hidden_size, size_linear_lyr)

        # Fully connected layers
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(2*size_linear_lyr, num_classes)
    
    def forward(self, x):
        # Input shape: (batchsamples, time, channels)
        xlstm = x
        xcnn = x.permute(0, 2, 1)  # (batch_size, input_channels, timesteps)
        
        # CNN
        for i in range(self.num_conv_layers):
            xcnn = self.conv_layers[i](xcnn)
            xcnn = self.bn_layers[i](xcnn)    # Batch Normalization
            xcnn = F.relu(xcnn) 
            xcnn = self.pool_layers[i](xcnn)

        # Flatten the output
        xcnn = self.flatten(xcnn)
        # Fully connected layer CNN
        xcnn = self.fc_cnn(xcnn)
        xcnn = F.relu(xcnn)
        
        # LSTM
        batch_size = xlstm.size(0)
        h0 = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_size).to(x.device)  # Hidden state
        c0 = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_size).to(x.device)  # Cell state
        xlstm, _ = self.lstm(xlstm, (h0, c0))
        # Fully connected layer LSTM
        xlstm = F.relu(self.fc_lstm(xlstm[:, -1, :]))

        # Concatenate CNN and LSTM outputs
        x = torch.cat([xcnn, xlstm], dim=1)

        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# Residual Block for 1D CNN
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)  # Skip connection
        out = self.relu(out)
        
        return out

# 1D CNN Model with Configurable Skip Connections and Conv Layers
class CNNSkipConnectionsLSTM(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, num_layers=4, 
                 num_blocks_per_layer=2, initial_channels=32, 
                 num_lstm_layers=2, lstm_hidden_size=32, size_linear_lyr=16):
        super(CNNSkipConnectionsLSTM, self).__init__()
        
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        
        # Initial layers
        self.conv1 = nn.Conv1d(input_channels, initial_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(initial_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Dynamically create residual layers
        self.layers = nn.ModuleList()
        in_channels = initial_channels
        out_channels = initial_channels
        
        for i in range(num_layers):
            # Double the number of channels after the first layer
            if i > 0:
                out_channels *= 2
            
            # Create a layer with `num_blocks_per_layer` residual blocks
            self.layers.append(self._make_layer(in_channels, out_channels, num_blocks_per_layer, stride=2 if i > 0 else 1))
            
            # Update in_channels for the next layer
            in_channels = out_channels
        
        # Global average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # LSTM layers
        self.lstm = nn.LSTM(input_size=out_channels, hidden_size=lstm_hidden_size,
                            num_layers=num_lstm_layers, 
                            batch_first=True, dropout=0.2)

        # Fully connected layers
        self.fc1 = nn.Linear(lstm_hidden_size, size_linear_lyr)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(size_linear_lyr, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride=stride))  # First block with potential downsampling
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))  # Subsequent blocks
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Input shape: (batch-samples, time, channels)
        x = x.permute(0, 2, 1)  # (batch_size, input_channels, timesteps)
        
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual layers
        for layer in self.layers:
            x = layer(x)
        
        # Global average pooling and fully connected layer
        x = self.avgpool(x)  # Global average pooling

        # LSTM layers
        h0 = torch.zeros(self.num_lstm_layers, x.size(0), self.lstm_hidden_size).to(x.device)  # Hidden state
        c0 = torch.zeros(self.num_lstm_layers, x.size(0), self.lstm_hidden_size).to(x.device) # Cell state
        x = x.permute(0, 2, 1) # (batch_size, timesteps, channels
        x, _ = self.lstm(x,(h0,c0))

        # Fully connected layers
        x = F.relu(self.fc1(x[:, -1, :]))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    

class CNNSkipConnectionsLSTMParallel(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, num_layers=4, 
                 num_blocks_per_layer=2, initial_channels=32, 
                 num_lstm_layers=2, lstm_hidden_size=32, size_linear_lyr=16):
        super(CNNSkipConnectionsLSTMParallel, self).__init__()
        
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        
        # Initial layers
        self.conv1 = nn.Conv1d(input_channels, initial_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(initial_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Dynamically create residual layers
        self.layers = nn.ModuleList()
        in_channels = initial_channels
        out_channels = initial_channels
        
        for i in range(num_layers):
            # Double the number of channels after the first layer
            if i > 0:
                out_channels *= 2
            
            # Create a layer with `num_blocks_per_layer` residual blocks
            self.layers.append(self._make_layer(in_channels, out_channels, num_blocks_per_layer, stride=2 if i > 0 else 1))
            
            # Update in_channels for the next layer
            in_channels = out_channels
        
        # Global average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc_cnn = nn.LazyLinear(out_features=size_linear_lyr)

        # LSTM layers
        self.lstm = nn.LSTM(input_size=input_channels, hidden_size=lstm_hidden_size,
                            num_layers=num_lstm_layers, 
                            batch_first=True, dropout=0.2)

        # Fully connected layers
        self.fc_lstm = nn.Linear(lstm_hidden_size, size_linear_lyr)



        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(2*size_linear_lyr, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride=stride))  # First block with potential downsampling
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))  # Subsequent blocks
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Input shape: (batch-samples, time, channels)
        xlstm = x
        xcnn = x.permute(0, 2, 1)  # (batch_size, input_channels, timesteps)
        
        # Initial layers
        xcnn = self.conv1(xcnn)
        xcnn = self.bn1(xcnn)
        xcnn = self.relu(xcnn)
        xcnn = self.maxpool(xcnn)
        
        # Residual layers
        for layer in self.layers:
            xcnn = layer(xcnn)
        
        # Global average pooling and fully connected layer
        xcnn = self.avgpool(xcnn)  # Global average pooling
        xcnn = self.flatten(xcnn)
        # Fully connected layer CNN
        xcnn = self.fc_cnn(xcnn)
        xcnn = F.relu(xcnn)

        # LSTM layers
        h0 = torch.zeros(self.num_lstm_layers, x.size(0), self.lstm_hidden_size).to(x.device)  # Hidden state
        c0 = torch.zeros(self.num_lstm_layers, x.size(0), self.lstm_hidden_size).to(x.device) # Cell state
        xlstm, _ = self.lstm(xlstm,(h0,c0))

        # Fully connected layers
        xlstm = F.relu(self.fc_lstm(xlstm[:, -1, :]))

        # Concatenate CNN and LSTM outputs
        x= torch.cat([xcnn, xlstm], dim=1)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    