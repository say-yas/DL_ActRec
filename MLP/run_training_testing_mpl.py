
import time

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

import MLP.nn_mlp_model as mlp
import MLP.nn_mlp_train as mlp_train

def create_increasing_decreasing_list(n, initial_size, final_size):
  """Creates a list with first increasing and then decreasing values.

  Args:
    n: The length of the list.

  Returns:
    A list with first increasing and then decreasing values.
  """

  increasing_part = list(range(initial_size, n // 2 + initial_size))
  decreasing_part = list(range(final_size + (n // 2), final_size, -1))
  return increasing_part + decreasing_part

def run_mlp_training(data, labels, classes, device, n_features=6, n_classes=2, test_size=0.3,
                      val_size=0.1, batch_size=32, num_cpus=1,
                      lr=0.001, num_epochs=100,  patience=5, modeltype = "mlp1",
                        num_hidden_lyr=2, hidden_lyr_size=100, opt="adamW", verbose=False, pathsave="./", weights=None):
    
    """
    Runs the training of a MLP model
    Args:
    data: pd.DataFrame
        input data  
    labels: pd.DataFrame
        output data
    device: torch.device
        device to run the training on
    n_features: int
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
    hidden_lyr_size: int
        number of hidden layers in the network
    verbose: bool
        Print additional information. Defaults to False.
    Returns:
    computation_time: float
        time it took to train the model
    """
    
    parameters = {"test_size": test_size, "val_size": val_size,
                   "batch_size": batch_size, "num_cpus": num_cpus, "modeltype": modeltype,
                "lr": lr, "num_epochs": num_epochs, "verbose": verbose, "n_features": n_features,
                  "n_classes": n_classes, "patience": patience, "num_hidden_lyr": num_hidden_lyr,
                      "hidden_lyr_size": hidden_lyr_size, "opt": opt, "weights": weights}
    
    print("parameters: ", parameters)
    start = time.time()
    print("training started")
    # create model
    num_input, num_output = (n_features, n_classes)
    if modeltype =="mlp1":
        num_hidden, num_layers = (hidden_lyr_size, num_hidden_lyr) 
        model = mlp.MLPI(num_input, num_output, num_hidden, num_layers)
    else:
        hidden_lyrs = create_increasing_decreasing_list(num_hidden_lyr, hidden_lyr_size, 1+hidden_lyr_size)
        model = mlp.MLPII(num_input, num_output, hidden_lyrs)
    print(summary(model, input_size=(batch_size, n_features)))

    # Initialize the weights using Kaiming/Xavier initialization
    mlp.init_weights(model, "kaiming")

    # move model to device
    model=model.to(device)
    # create optimizer, loss function, early stopper and scheduler
    optimizer=optim.Adam(model.parameters(),lr=lr, weight_decay=1e-4) if opt=="adam" else optim.AdamW(model.parameters(),lr=lr, weight_decay=1e-4)
    if weights is not None:
        loss_function=nn.CrossEntropyLoss(weight=weights.to(device))
    else: 
        loss_function=nn.CrossEntropyLoss()
    early_stopper = mlp.EarlyStopper(verbose=verbose, path="checkpoint.pt", patience=patience)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                                    factor=0.9, patience=patience-4,
                                                                        threshold=1e-8
                                                                        # , verbose=verbose #verbose is deprecated
                                                                        )

    # create training object
    train_model = mlp_train.TrainMLP(model, optimizer, loss_function, early_stopper, scheduler, device, parameters)
    x_train, x_val, x_test, y_train, y_val, y_test = train_model.generate_train_val_test(data, labels)

    xy_train = mlp_train.data_to_tensor(x_train, y_train)
    xy_val = mlp_train.data_to_tensor(x_val, y_val)
    xy_test = mlp_train.data_to_tensor(x_test, y_test)


    # create data loaders
    trainloader, valloader, testloader = train_model.init_data_loaders(xy_train, xy_val, xy_test)

    # train model
    train_losses, val_losses, train_accs, val_accs, confusion_matrix, train_f1, val_f1 = train_model.run_training(trainloader, valloader)

    # print accuracy and loss
    strcomplexity = f"MLP2, [{hidden_lyrs}]: " if modeltype == "mlp2" else f"MLP1, ({num_hidden_lyr} , {hidden_lyr_size}): "
    mlp_train.plot_progress(strcomplexity+"loss","loss", train_losses, val_losses, save_path=pathsave+"loss.png")
    mlp_train.plot_progress(strcomplexity+"Accuracy","Accuracy", train_accs, val_accs, save_path=pathsave+"accuracy.png")
    mlp_train.plot_progress(strcomplexity+"F1score","F1score", train_f1, val_f1, save_path=pathsave+"f1score.png")



    # test model
    correct, total, accuracy_test, f1score_test = train_model.testing_step(testloader)
    cm = train_model.get_confusion_matrix(testloader, view_cm=False)
    mlp_train.visualize_confusion_matrix(cm, classes, correct, total, path=pathsave+"confusion_matrix_testloader.png")
    print(f"Accuracy on test set: {accuracy_test:.4f}")
    print(f"F1score on test set: {f1score_test:.4f}")

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
    
    