import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import Transformer.nn_Relative_positionalembedding as rel_embd


def init_weights(m, init_type="kaiming"):
    if isinstance(m, nn.Linear):
        if init_type=="kaiming":
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif init_type=="xavier":
            nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)



# Positional Encoding - https://github.com/BrandenKeck/pytorch_fun/blob/main/timeseries_transformer/forecasting_model.py
# Copied from PyTorch Library https://github.com/pytorch/pytorch/issues/51551
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerI(torch.nn.Module):
    def __init__(self, 
                 input_channels = 3,
                 output_size = 5,
                 seq_len=200,
                 embed_size = 16,
                 nhead = 4,
                 dim_feedforward = 2048,
                 dropout = 0.0,
                 conv1d_emb = True,
                 conv1d_kernel_size = 3,
                 size_linear_layers = 16,
                 num_encoderlayers = 1,
                 device = None):
        """
        TransformerI Model for Time Series Forecasting
        Args:
            input_channels (int): Number of input channels. Defaults to 3.
            output_size (int): Number of output channels. Defaults to 5.
            seq_len (int): Length of the input sequence. Defaults to 200.
            embed_size (int): Size of the embedding. Defaults to 16.
            nhead (int): Number of attention heads. Defaults to 4.
            dim_feedforward (int): Dimension of the feedforward network model. Defaults to 2048.
            dropout (float): Dropout rate. Defaults to 0.0.
            conv1d_emb (bool): Use Conv1D for embedding. Defaults to True.
            conv1d_kernel_size (int): Kernel size for Conv1D embedding. Defaults to 3.
            size_linear_layers (int): Size of the linear layers. Defaults to 16.
            num_encoderlayers (int): Number of encoder layers in the transformer. Defaults to 1.
            device: Device on which the model will be run. Defaults to None.
        """
        super(TransformerI, self).__init__()

        # Set Class-level Parameters
        self.conv1d_emb = conv1d_emb
        self.conv1d_kernel_size = conv1d_kernel_size
        self.seq_len = seq_len
        self.embed_size = embed_size
        self.n_heads = nhead
        self.device = device
        
        assert embed_size % nhead == 0
        
        self.hid_dim = embed_size
        self.head_dim = embed_size // nhead
        self.max_relative_position = 2

        # Input Embedding Component
        if conv1d_emb:
            if conv1d_kernel_size%2==0:
                raise Exception("conv1d_kernel_size must be an odd number to preserve dimensions.")
            self.conv1d_padding = int( (conv1d_kernel_size - 1)/2 )
            self.input_embedding  = nn.Conv1d(input_channels, embed_size, 
                                              kernel_size=conv1d_kernel_size, 
                                              padding=self.conv1d_padding)
        else: self.input_embedding  = nn.Linear(input_channels, embed_size)

        # Positional Encoder Componet (See Code Copied from PyTorch Above)
        self.position_encoder = PositionalEncoding(d_model=embed_size, 
                                                dropout=dropout,
                                                max_len=seq_len)
        
        # Transformer Encoder Layer Component
        self.transformer_encoder = nn.modules.transformer.TransformerEncoderLayer(
            d_model = embed_size,
            nhead = self.n_heads,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            batch_first = True
        )
        self.transformer = nn.TransformerEncoder(self.transformer_encoder, num_layers=num_encoderlayers)

        
        # Regression Component
        self.linear1 = nn.Linear(seq_len*embed_size, int(dim_feedforward))
        self.linear2 = nn.Linear(int(dim_feedforward), int(size_linear_layers))
        self.outlayer = nn.Linear(int(size_linear_layers), output_size)

        # Basic Components
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    # Model Forward Pass
    def forward(self, x):
        src_mask = self._generate_square_subsequent_mask()
        src_mask.to(self.device)
        if self.conv1d_emb: 
            x = x.permute(0, 2, 1)
            # x = F.pad(x, (0, 0, self.conv1d_padding, 0), "constant", -1)
            x = self.input_embedding(x)
            x = x.permute(0, 2, 1)
        else: 
            x = self.input_embedding(x)

        x = x.permute(1, 0, 2)
        x = self.position_encoder(x)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x, src_mask=src_mask)
        x = self.transformer(x)
        x = x.reshape((-1, self.seq_len*self.embed_size))
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.outlayer(x)
    
    # Function Copied from PyTorch Library to create upper-triangular source mask
    def _generate_square_subsequent_mask(self):
        return torch.triu(
            torch.full((self.seq_len, self.seq_len), float('-inf'), dtype=torch.float32),
            diagonal=1,
        )
    

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
    