import torch.nn as nn
import torch
import numpy as np

def init_weights(m, init_type="kaiming"):
    if isinstance(m, nn.Linear):
        if init_type=="kaiming":
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif init_type=="xavier":
            nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)

class MLPI(nn.Module):
    "Defines a standard fully-connected network in PyTorch"
    """
    Args:  
    num_input: int
        number of input features
    num_output: int
        number of output features
    num_hidden: int
        number of hidden units in the network
    num_layers: int
        number of hidden layers in the network 
    """
    
    def __init__(self, num_input, num_output, num_hidden, num_layers):
        super().__init__()
        activation = nn.ReLU
        self.fcs = nn.Sequential(*[
                        nn.Linear(num_input, num_hidden),
                        activation()])
        self.dropout1 = nn.Dropout(0.2)
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(num_hidden, num_hidden),
                            nn.Dropout(0.3),
                            activation()]) for _ in range(num_layers-1)])
        self.dropout2 = nn.Dropout(0.2)
        self.fce = nn.Linear(num_hidden, num_output)

    def forward(self, x):
        x = self.fcs(x)
        x = self.dropout1(x)
        x = self.fch(x)
        x = self.dropout2(x)
        x = self.fce(x)
        return x
    
class MLPII(nn.Module):
    "Defines a standard fully-connected network in PyTorch"
    """
    Args:  
    num_input: int
        number of input features
    num_output: int
        number of output features
    hidden_lyrs: list
        list of size of hidden units in the network 
    """
    
    def __init__(self, num_input, num_output, hidden_lyrs):
        super().__init__()
                               
        activation = nn.ReLU
        self.fcs = nn.Sequential(*[
                        nn.Linear(num_input, hidden_lyrs[0]),
                        activation()])
        self.dropout1 = nn.Dropout(0.2)
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(hidden_lyrs[idx], hidden_lyrs[idx+1]),
                            nn.Dropout(0.3),
                            activation()]) for idx,_ in enumerate(hidden_lyrs[:-1])])
        self.dropout2 = nn.Dropout(0.2)
        self.fce = nn.Linear(hidden_lyrs[-1], num_output)

    def forward(self, x):
        x = self.fcs(x)
        x = self.dropout1(x)
        x = self.fch(x)
        x = self.dropout2(x)
        x = self.fce(x)
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
    