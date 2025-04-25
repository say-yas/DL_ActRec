
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

import CNN_LSTM.nn_CNN_model as cnn
import CNN_LSTM.nn_LSTM_model as lstm
import CNN_LSTM.nn_CNN_LSTM_model as cnn_lstm

import CNN_LSTM.nn_CNN_train as cnn_train


def run_cnnlstm_training(data, labels, classes, device, num_channels=6, n_classes=2, test_size=0.3,
                      val_size=0.1, batch_size=32, num_cpus=1,
                      lr=0.001, num_epochs=100,  patience=5, modeltype = "cnn1",
                        max_length_series=2, num_conv_layers=100, size_linear_lyr=10, 
                        num_blocks_per_layer=2, initial_channels=32,
                        lstm_hidden_size=32, num_lstm_layers=1,
                        opt="adamW", verbose=False, pathsave="./", weights=None, norm_type="per-channel"):
    
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
    num_hidden_lyr: int
        number of hidden units in the network
    num_conv_layers: int
        number of convolutional layers in the network
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
                  "size_linear_lyr": size_linear_lyr,"num_blocks_per_layer": num_blocks_per_layer,
                  "initial_channels": initial_channels,"num_conv_lyr": num_conv_layers, 
                  "lstm_hidden_size": lstm_hidden_size, "num_lstm_layers": num_lstm_layers,
                  "opt": opt, "weights": weights,
                  "norm_type": norm_type}
    
    print("parameters: ", parameters)
    start = time.time()
    print("training started")
    # create model
    num_input, num_output = (num_channels, n_classes)
    if modeltype =="cnn1":
        model = cnn.CNNI(num_input, num_output, max_length_series,
                          num_conv_layers, size_linear_lyr, initial_channels)
        input_shape = (max_length_series, num_channels)
        print(summary(model, input_size=input_shape))
    elif modeltype == "cnnskip":
        model = cnn.CNNSkipConnections(num_input, num_output, num_conv_layers,
                                        num_blocks_per_layer, initial_channels)
        input_shape = (max_length_series, num_channels)
        print(summary(model, input_size=input_shape))
    elif modeltype == "cnnbatchnorm":
        model = cnn.CNNwithBatchNorm(num_input, num_output, max_length_series,
                                      num_conv_layers, size_linear_lyr, initial_channels)
        input_shape = (max_length_series, num_channels)
        print(summary(model, input_size=input_shape))
    elif modeltype == "lstm":
        model = lstm.LSTMI(num_input, num_output, max_length_series,
                           num_lstm_layers, lstm_hidden_size, size_linear_lyr)
        input_shape = (max_length_series, num_channels)
        print(torchinfo.summary(model, input_size=input_shape, batch_dim = 0))
    elif modeltype == "cnnlstm1":
        model = cnn_lstm.CNN_LSTMI(num_input, num_output, max_length_series, 
                                   num_conv_layers, initial_channels, 
                                   num_lstm_layers, lstm_hidden_size, 
                                   size_linear_lyr)
        input_shape = (max_length_series, num_channels)
        print(torchinfo.summary(model, input_size=input_shape, batch_dim = 0))
    elif modeltype == "cnnlstmskip":
        model = cnn_lstm.CNNSkipConnectionsLSTM(num_input, num_output, num_conv_layers,
                                        num_blocks_per_layer, initial_channels, num_lstm_layers, 
                                        lstm_hidden_size, size_linear_lyr)
        input_shape = (max_length_series, num_channels)
        print(torchinfo.summary(model, input_size=input_shape, batch_dim = 0))
    elif modeltype == "cnnlstmbatchnorm":
        model = cnn_lstm.CNNwithBatchNormLSTM(num_input, num_output, 
                                     num_conv_layers, size_linear_lyr, initial_channels, 
                                     num_lstm_layers, lstm_hidden_size) 
        input_shape = (max_length_series, num_channels)
        print(torchinfo.summary(model, input_size=input_shape, batch_dim = 0))
    elif modeltype == "cnnlstmbatchnormparallel":
        model = cnn_lstm.CNNwithBatchNormLSTMParrallel(num_input, num_output,
                                                       num_conv_layers, size_linear_lyr, 
                                                       initial_channels, num_lstm_layers, 
                                                       lstm_hidden_size)
        input_shape = (max_length_series, num_channels)
        print(torchinfo.summary(model, input_size=input_shape, batch_dim = 0))
    elif modeltype == "cnnlstmskipparallel":
        model = cnn_lstm.CNNSkipConnectionsLSTMParallel(num_input, num_output, num_conv_layers,
                                        num_blocks_per_layer, initial_channels, num_lstm_layers, 
                                        lstm_hidden_size, size_linear_lyr)
        input_shape = (max_length_series, num_channels)
        print(torchinfo.summary(model, input_size=input_shape, batch_dim = 0))


    print(batch_size, max_length_series, num_channels)
    

    # Initialize the weights using Kaiming/Xavier initialization
    cnn.init_weights(model, "kaiming")

    # move model to device
    model=model.to(device)
    # create optimizer, loss function, early stopper and scheduler
    optimizer=optim.Adam(model.parameters(),lr=lr, weight_decay=1e-4) if opt=="adam" else optim.AdamW(model.parameters(),lr=lr, weight_decay=1e-4)
    if weights is not None:
        loss_function=nn.CrossEntropyLoss(weight=weights.to(device))
    else: 
        loss_function=nn.CrossEntropyLoss()
    early_stopper = cnn.EarlyStopper(verbose=verbose, path="checkpoint.pt", patience=patience)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                                    factor=0.9, patience=patience-4,
                                                                        threshold=1e-8
                                                                        # , verbose=verbose #verbose is deprecated
                                                                        )

    # create training object
    train_model = cnn_train.TrainCNN(model, optimizer, loss_function, early_stopper, scheduler, device, parameters)
    x_train, x_val, x_test, y_train, y_val, y_test = train_model.generate_train_val_test(data, labels)

    print("x_train shape: ", np.array(x_train).shape, "y_train shape: ", np.array(y_train).shape)
    xy_train = cnn_train.data_to_tensor(x_train, y_train)
    print("x_val shape: ", np.array(x_val).shape, "y_val shape: ", np.array(y_val).shape)
    xy_val = cnn_train.data_to_tensor(x_val, y_val)
    print("x_test shape: ", np.array(x_test).shape, "y_test shape: ", np.array(y_test).shape)
    xy_test = cnn_train.data_to_tensor(x_test, y_test)


    # create data loaders
    trainloader, valloader, testloader = train_model.init_data_loaders(xy_train, xy_val, xy_test)

    # train model
    train_losses, val_losses, train_accs, val_accs, confusion_matrix, train_f1, val_f1, train_gmean, val_gmean, train_pre, val_pre, train_rec, val_rec = train_model.run_training(trainloader,
                                                                                                                                                                                  valloader)


    # print accuracy and loss
    if modeltype =="cnn1":
        strcomplexity = f"CNNI, [{num_input, num_output, max_length_series,num_conv_layers, size_linear_lyr, initial_channels}]: "
    elif modeltype == "cnnskip":
        strcomplexity = f"CNNskip, [{num_input, num_output, num_conv_layers, num_blocks_per_layer, initial_channels}]: "
    elif modeltype =="cnnbatchnorm":
        strcomplexity = f"CNNwithBatchNorm, [{num_input, num_output, max_length_series, num_conv_layers, size_linear_lyr, initial_channels}]: "
    elif modeltype == "lstm":
        strcomplexity = f"LSTM, [{num_input, num_output, max_length_series, num_lstm_layers, lstm_hidden_size, size_linear_lyr}]: "
    elif modeltype == "cnnlstm1":
        strcomplexity = f"CNN-LSTMI, [{num_input, num_output, max_length_series, num_conv_layers,initial_channels, num_lstm_layers, lstm_hidden_size, size_linear_lyr}]: "
    elif modeltype == "cnnlstmskip":
        strcomplexity = f"CNN-LSTMskip, [{num_input, num_output, num_conv_layers,num_blocks_per_layer, initial_channels, num_lstm_layers, lstm_hidden_size, size_linear_lyr}]: "
    elif modeltype == "cnnlstmbatchnorm":
        strcomplexity = f"CNN-LSTMwithBatchNorm, [{num_input, num_output, num_conv_layers, size_linear_lyr, initial_channels, num_lstm_layers, lstm_hidden_size}]: "
    elif modeltype == "cnnlstmbatchnormparallel":
        strcomplexity = f"CNN-LSTMwithBatchNormParrallel, [{num_input, num_output,num_conv_layers, size_linear_lyr, initial_channels, num_lstm_layers, lstm_hidden_size}]: "
    elif modeltype == "cnnlstmskipparallel":
        strcomplexity = f"CNN-LSTMskipParallel, [{num_input, num_output, num_conv_layers,num_blocks_per_layer, initial_channels, num_lstm_layers, lstm_hidden_size, size_linear_lyr}]: "


    cnn_train.plot_progress(strcomplexity+"loss","loss", train_losses, val_losses, save_path=pathsave+"loss.png")
    cnn_train.plot_progress(strcomplexity+"Accuracy","Accuracy", train_accs, val_accs, save_path=pathsave+"accuracy.png")
    cnn_train.plot_progress(strcomplexity+"F1score","F1score", train_f1, val_f1, save_path=pathsave+"f1score.png")
    cnn_train.plot_progress(strcomplexity+"Gmean","Gmean", train_gmean, val_gmean, save_path=pathsave+"gmean.png")
    cnn_train.plot_progress(strcomplexity+"Precision","Precision", train_pre, val_pre, save_path=pathsave+"precision.png")
    cnn_train.plot_progress(strcomplexity+"Recall","Recall", train_rec, val_rec, save_path=pathsave+"recall.png")



    # test model
    correct, total, accuracy_test, f1score_test, gmean_test, precision_test, recall_test = train_model.testing_step(testloader)
    cm = train_model.get_confusion_matrix(testloader, view_cm=False)
    cnn_train.visualize_confusion_matrix(cm, classes, correct, total, path=pathsave+"confusion_matrix_testloader.png")
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
    
    