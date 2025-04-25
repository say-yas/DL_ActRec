
import time

import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import torchinfo

import Transformer.nn_Transformer_model as transformer

import Transformer.nn_Transformer_train as transformer_train


def run_transformer_training(data, labels, classes, device, num_channels=6, n_classes=2, test_size=0.3,
                      val_size=0.1, batch_size=32, num_cpus=1,
                      lr=0.001, num_epochs=100,  patience=5, modeltype = "trans1",
                        max_length_series=2, embed_size = 16,
                        nhead = 4,dim_feedforward = 2048,  num_encoderlayers = 1,
                        dropout = 0.0,conv1d_emb = True, 
                        conv1d_kernel_size = 3, size_linear_layers = 16,
                        opt="adamW", verbose=False, pathsave="./", weights=None, norm_type="per-channel", num_training=1):
    
    """
    Runs the training of a MLP model
    Args:
    data: pd.DataFrame
        input data  
    labels: pd.DataFrame
        output data
    device: torch.device
        device to run the training on
    num_channels: int
        number of input features
    n_classes: int
        number of output features
    test_size: float    
        size of the test set
    val_size: float 
        size of the validation set
    batch_size: int
        size of the batch
    num_cpus: int
        number of cpus to use
    lr: float
        learning rate
    num_epochs: int
        number of epochs
    patience: int
        number of epochs to wait for decreasing loss. If loss does not increase, stop training early.
    modeltype: str
        type of model to use. Options are: transfer
    nhead: int
        number of heads in the multihead attention layer
    dim_feedforward: int
        dimension of the feedforward layer
    dropout: float
        dropout rate
    conv1d_emb: bool
        if True, use 1D convolutional embedding
    conv1d_kernel_size: int
        kernel size of the 1D convolutional embedding
    size_linear_layers: int
        size of the linear layers
    num_encoderlayers: int
        number of encoder layers
    opt: str
        optimizer to use. Options are: adam, adamW
    num_training: int
        number of training runs
    verbose: bool
        Print additional information. Defaults to False.
    Returns:
    computation_time: float
        time it took to train the model
    """
    
    parameters = {"test_size": test_size, "val_size": val_size,
                   "batch_size": batch_size, "num_cpus": num_cpus, "modeltype": modeltype,
                  "lr": lr, "num_epochs": num_epochs, "verbose": verbose, "num_channels": num_channels,
                  "n_classes": n_classes, "patience": patience, "max_length_series": max_length_series,
                     "nhead": nhead, "dim_feedforward": dim_feedforward, "embed_size": embed_size,
                    "dropout": dropout, "conv1d_emb": conv1d_emb, "conv1d_kernel_size": conv1d_kernel_size,
                    "size_linear_layers": size_linear_layers, "num_encoderlayers": num_encoderlayers,
                  "opt": opt, "weights": weights,
                  "norm_type": norm_type, "num_training": num_training}
    
    print("parameters: ", parameters)
    start = time.time()
    print("training started")
    # create model
    num_input, num_output = (num_channels, n_classes)
    if modeltype =="trans1":
        model = transformer.TransformerI(input_channels = num_channels,
                                         output_size = n_classes,
                 seq_len=max_length_series,
                 embed_size = embed_size,
                 nhead = nhead,
                 dim_feedforward = dim_feedforward,
                 dropout = dropout,
                 conv1d_emb = conv1d_emb,
                 conv1d_kernel_size = conv1d_kernel_size,
                 size_linear_layers = size_linear_layers,
                 num_encoderlayers = num_encoderlayers,
                  device=device)
        input_shape = (max_length_series, num_channels)
        print(torchinfo.summary(model, input_size=input_shape, batch_dim = 0))


    print(batch_size, max_length_series, num_channels)
    

    # Initialize the weights using Kaiming/Xavier initialization
    transformer.init_weights(model, "kaiming")

    # move model to device
    model=model.to(device)
    # create optimizer, loss function, early stopper and scheduler
    optimizer=optim.Adam(model.parameters(),lr=lr, weight_decay=1e-4) if opt=="adam" else optim.AdamW(model.parameters(),lr=lr, weight_decay=1e-4)
    if weights is not None:
        loss_function=nn.CrossEntropyLoss(weight=weights.to(device))
    else: 
        loss_function=nn.CrossEntropyLoss()
    early_stopper = transformer.EarlyStopper(verbose=verbose, path="checkpoint.pt", patience=patience)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                                    factor=0.9, patience=patience-4,
                                                                        threshold=1e-8
                                                                        # , verbose=verbose #verbose is deprecated
                                                                        )

    # create training object
    train_model = transformer_train.TrainTransformer(model, optimizer, loss_function, early_stopper, scheduler, device, parameters)
    x_train, x_val, x_test, y_train, y_val, y_test = train_model.generate_train_val_test(data, labels)

    print("x_train shape: ", np.array(x_train).shape, "y_train shape: ", np.array(y_train).shape)
    xy_train = transformer_train.data_to_tensor(x_train, y_train)
    print("x_val shape: ", np.array(x_val).shape, "y_val shape: ", np.array(y_val).shape)
    xy_val = transformer_train.data_to_tensor(x_val, y_val)
    print("x_test shape: ", np.array(x_test).shape, "y_test shape: ", np.array(y_test).shape)
    xy_test = transformer_train.data_to_tensor(x_test, y_test)


    # create data loaders
    trainloader, valloader, testloader = train_model.init_data_loaders(xy_train, xy_val, xy_test)

    # train model
    train_losses, val_losses, train_accs, val_accs, confusion_matrix, train_f1, val_f1, train_gmean, val_gmean, train_pre, val_pre, train_rec, val_rec = train_model.run_training(trainloader,
                                                                                                                                                                                  valloader)


    # print accuracy and loss
    if modeltype =="trans1":
        strcomplexity = f"TransformerI, [{num_input, num_output, max_length_series, nhead, dim_feedforward, num_encoderlayers, conv1d_emb, conv1d_kernel_size, size_linear_layers }]: "


    transformer_train.plot_progress(strcomplexity+"loss","loss", train_losses, val_losses, save_path=pathsave+"loss.png")
    transformer_train.plot_progress(strcomplexity+"Accuracy","Accuracy", train_accs, val_accs, save_path=pathsave+"accuracy.png")
    transformer_train.plot_progress(strcomplexity+"F1score","F1score", train_f1, val_f1, save_path=pathsave+"f1score.png")
    transformer_train.plot_progress(strcomplexity+"Gmean","Gmean", train_gmean, val_gmean, save_path=pathsave+"gmean.png")
    transformer_train.plot_progress(strcomplexity+"Precision","Precision", train_pre, val_pre, save_path=pathsave+"precision.png")
    transformer_train.plot_progress(strcomplexity+"Recall","Recall", train_rec, val_rec, save_path=pathsave+"recall.png")



    # test model
    correct, total, accuracy_test, f1score_test, gmean_test, precision_test, recall_test = train_model.testing_step(testloader)
    cm = train_model.get_confusion_matrix(testloader, view_cm=False)
    transformer_train.visualize_confusion_matrix(cm, classes, correct, total, path=pathsave+"confusion_matrix_testloader.png")
    print(f"Accuracy on test set: {accuracy_test:.4f}")
    print(f"F1score on test set: {f1score_test:.4f}")
    print(f"Gmean on test set: {gmean_test:.4f}")
    print(f"Precision on test set: {precision_test:.4f}")
    print(f"Recall on test set: {recall_test:.4f}")

    cm_val = confusion_matrix.numpy()
    if verbose:
        df_cm = pd.DataFrame(cm_val, index = [i for i in classes],
                  columns = [i for i in classes])
        fig, ax = plt.subplots(figsize=(12, 10))
        sn.heatmap(df_cm/ (df_cm.astype("float").sum() + 1e-9), annot=True)
        ax.set_title(strcomplexity)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        # plt.show()
        ax.figure.savefig(pathsave+"confusion_matrix_test_allnormalized.png")

    end = time.time()
    computation_time = end - start
    print(f"computation time: {end-start:.4f}")

    return computation_time
    
    