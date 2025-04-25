import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def init_weights(m, init_type="kaiming"):
    """Initialize weights of the model.
    
    Args:
        m (nn.Module): Model layer to initialize.
        init_type (str, optional): Type of initialization.Defaults to "kaiming"."
    """
    if isinstance(m, nn.Linear):
        if init_type=="kaiming":
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif init_type=="xavier":
            nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)

class CNNI(nn.Module):
    """1D Convolutional Neural Network with configurable number of layers and channels.
    
    Args:
        input_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        max_length_series (int): Maximum length of the input series.
        num_conv_layers (int, optional): Number of convolutional layers. Defaults to 2.
        size_linear_lyr (int, optional): Size of the linear layer. Defaults to 32.
        initial_channels (int, optional): Number of initial channels. Defaults to 8.
    """
    def __init__(self, input_channels, num_classes,  max_length_series, num_conv_layers=2, 
                 size_linear_lyr=32, initial_channels=8):
        super(CNNI, self).__init__()
        
        self.num_conv_layers = num_conv_layers
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        # Dynamically create convolutional and pooling layers
        in_channels = input_channels
        
        for i in range(num_conv_layers):
            self.conv_layers.append(nn.Conv1d(in_channels, initial_channels, kernel_size=3, padding=1))
            self.pool_layers.append(nn.MaxPool1d(kernel_size=2))
            
            # Update in_channels and out_channels for the next layer
            in_channels = initial_channels
            initial_channels *= 2  # Double the number of channels for the next layer
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Calculate the size of the flattened output
        self.fc_input_size = in_channels * (max_length_series // (2 ** num_conv_layers))
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, size_linear_lyr)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(size_linear_lyr, num_classes)
    
    def forward(self, x):
        # Input shape: (batchsamples, time, channels)
        x = x.permute(0, 2, 1)  # (batch_size, input_channels, timesteps)
        
        # Apply convolutional and pooling layers
        for i in range(self.num_conv_layers):
            x = F.relu(self.conv_layers[i](x))
            x = self.pool_layers[i](x)
        
        # Flatten the output
        x = self.flatten(x)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    

class CNNwithBatchNorm(nn.Module):
    """1D Convolutional Neural Network with Batch Normalization and configurable number of layers and channels.
    Args:
        input_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        max_length_series (int): Maximum length of the input series.
        num_conv_layers (int, optional): Number of convolutional layers. Defaults to 2.
        size_linear_lyr (int, optional): Size of the linear layer. Defaults to 32.
        initial_channels (int, optional): Number of initial channels. Defaults to 8.
    """
    def __init__(self, input_channels, num_classes,  max_length_series, num_conv_layers=2, 
                 size_linear_lyr=32, initial_channels=8):
        super(CNNwithBatchNorm, self).__init__()
        
        self.num_conv_layers = num_conv_layers
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
        
        # Calculate the size of the flattened output
        self.fc_input_size = in_channels * (max_length_series // (2 ** num_conv_layers))
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, size_linear_lyr)
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
        
        # Flatten the output
        x = self.flatten(x)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x



# Residual Block for 1D CNN
class ResidualBlock1D(nn.Module):
    """Residual Block for 1D Convolutional Neural Network.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        stride (int, optional): Stride for the convolution. Defaults to 1.
        padding (int, optional): Padding for the convolution. Defaults to 1.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock1D, self).__init__()
        
        # Convolutional layers
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
class CNNSkipConnections(nn.Module):
    """1D Convolutional Neural Network with Skip Connections and configurable number of layers and channels.
    Args:
        input_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        num_layers (int, optional): Number of residual layers. Defaults to 4.
        num_blocks_per_layer (int, optional): Number of residual blocks per layer. Defaults to 2.
        initial_channels (int, optional): Number of initial channels. Defaults to 32.
    """
    def __init__(self, input_channels=3, num_classes=10, num_layers=4, num_blocks_per_layer=2, initial_channels=32):
        super(CNNSkipConnections, self).__init__()
        
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
        self.fc = nn.Linear(out_channels, num_classes)  # Adjusted for final number of channels
    
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
        x = torch.flatten(x, 1)  # Flatten the tensor
        x = self.fc(x)  # Fully connected layer
        
        return x
    

class EarlyStopper:
    """Early stops the training if validation loss does not increase after a
    given patience.
    """
    def __init__(self, verbose=False, path='checkpoint.pt', patience=1):
        """Initialization.
        Args:
            verbose (bool, optional): Print additional information. Defaults to False.
            path (str, optional): Path where checkpoints should be saved. 
                Defaults to 'checkpoint.pt'.
            patience (int, optional): Number of epochs to wait for decreasing
                loss. If lossyracy does not increase, stop training early. 
                Defaults to 1.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.__early_stop = False
        self.val_loss_min = np.inf
        self.path = path
                
    @property
    def early_stop(self):
        """True if early stopping criterion is reached.
        Returns:
            [bool]: True if early stopping criterion is reached.
        """
        return self.__early_stop
                
    def update(self, val_loss, model):
        """Call after one epoch of model training to update early stopper object.
        Args:
            val_loss (float): lossuracy on validation set
            model (nn.Module): torch model that is trained
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model, val_loss)
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.__early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model, val_loss)
            self.counter = 0
            
    def save_checkpoint(self, model, val_loss):
        """Save model checkpoint.
        Args:
            model (nn.Module): Model of which parameters should be saved.
        """
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss        
        
    def load_checkpoint(self, model):
        """Load model from checkpoint.
        Args:
            model (nn.Module): Model that should be reset to parameters loaded
                from checkpoint.
        Returns:
            nn.Module: Model with parameters from checkpoint
        """
        model.load_state_dict(torch.load(self.path,  weights_only=True)) #map_location=device,

        return model
    