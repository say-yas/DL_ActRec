import os
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torcheval.metrics.functional import multiclass_f1_score, multiclass_recall, multiclass_precision, multilabel_auprc
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler


import fastprogress



class TrainCNN:
    def __init__(self, model, optimizer, criterion, early_stopper, scheduler, device, params):

        """
        model: nn.Module
            Model to be trained
        optimizer: torch.optim
            Optimizer to be used for training
        criterion: torch.nn
            Loss function to be used for training
        early_stopper: EarlyStopper
            Object to stop training early
        scheduler: torch.optim.lr_scheduler
            Learning rate scheduler
        device: torch.device
            Device to be used for training
        params: dict
            Dictionary containing parameters for training
        """

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.early_stopper = early_stopper
        self.scheduler = scheduler
        self.device = device
        self.params = params
        
    
    def generate_train_val_test(self, x, y):
        """
        Generate train, validation and test set.
        x: np.array
            Features
        y: np.array
            labels
        return: multiple np.array
            Train, validation and test datasets and labels
        """

        test_size = self.params["test_size"]
        val_size = self.params["val_size"]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=43,
            stratify=y, #Stratified sampling
              shuffle=True)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=val_size, random_state=43,
             stratify=y_train, #Stratified sampling
              shuffle=True)
        
        print("counts classes in train:", pd.Series(y_train).value_counts(), "in validation:", pd.Series(y_val).value_counts(), "in test:", pd.Series(y_test).value_counts())


        norm_type = self.params["norm_type"]
        x_train_scaled, x_val_scaled, x_test_scaled = normalize_data(x_train, x_test, x_val, norm_type)

        

        print("shapes:", x_train_scaled.shape, x_val_scaled.shape, x_test_scaled.shape)

        return x_train_scaled, x_val_scaled, x_test_scaled, y_train, y_val, y_test
    
    def init_data_loaders(self, trainset, valset, testset):
        """Initialize train, validation and test data loader.

        Args:
            trainset: pytorch Dataset object for training.
            valset: pytorch Dataset object for validation.
            testset: pytorch Dataset object for testing.
            batch_size (int): Batch size for training, validation and testing.
            num_cpus (int): Number of cpus to use for data loading.

        Returns:
            DataLoader, DataLoader, DataLoader: Returns pytorch DataLoader objects
                for training, validation and testing.
        """
        batch_size = self.params["batch_size"]
        num_cpus = self.params["num_cpus"]


        trainloader = torch.utils.data.DataLoader(trainset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_cpus
                                                )
        valloader = torch.utils.data.DataLoader(valset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_cpus
                                                )
        testloader = torch.utils.data.DataLoader(testset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_cpus
                                                )
        return trainloader, valloader, testloader
    

    def training_step(self, dataloader, master_bar):
        """Run one training epoch.

        Args:
            dataloader (DataLoader): Torch DataLoader object to load data
            master_bar (fastprogress.master_bar): Will be iterated over for each
                epoch to draw batches and display training progress

        Returns:
            float, float: Mean loss of this epoch, fraction of correct predictions
                on training set (accuracy)
        """

        epoch_loss = []
        epoch_y = []
        epoch_ypred = []
        epoch_correct, epoch_total = 0, 0

        for x, y in fastprogress.progress_bar(dataloader, parent=master_bar):
            self.optimizer.zero_grad()
            self.model.train()

            # Forward pass
            y_pred = self.model(x.to(self.device))

            # save the true and predicted labels
            epoch_y.append(y.tolist()) #check whetehr to be sent to device
            epoch_ypred.append(y_pred.argmax(dim=1).cpu().tolist())

            # save the number of correctly classified
            epoch_correct += sum(y.to(self.device) == y_pred.argmax(dim=1))
            epoch_total += len(y)

            # Compute loss
            loss = self.criterion(y_pred, y.to(self.device))

            # Backward pass
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2) # clip gradients
            self.optimizer.step()

            # For plotting the train loss, save it for each sample
            epoch_loss.append(loss.item())

        epoch_y = sum(epoch_y, [])
        epoch_ypred = sum(epoch_ypred, [])
        epoch_y = np.array(epoch_y).flatten()
        epoch_ypred = np.array(epoch_ypred).flatten()
        epoch_y = torch.tensor(epoch_y)
        epoch_ypred = torch.tensor(epoch_ypred)
        # compute f1score
        f1score = multiclass_f1_score(epoch_y, epoch_ypred, num_classes=self.params["n_classes"], average="macro")
        # compute recall
        recall = multiclass_recall(epoch_y, epoch_ypred, num_classes=self.params["n_classes"], average="macro")
        # compute precision
        precision = multiclass_precision(epoch_y, epoch_ypred, num_classes=self.params["n_classes"], average="macro")
        #gmean 
        gmean = weighted_geometric_mean_score(epoch_y, epoch_ypred)


        # Return the mean loss and the accuracy of this epoch
        return np.mean(epoch_loss), accuracy(epoch_correct, epoch_total), float(f1score.cpu()), float(precision.cpu()), float(recall.cpu()), float(gmean)
    
    def validatation_step(self, dataloader, master_bar):
        """Compute loss, accuracy and confusion matrix on validation set.

        Args:
            dataloader (DataLoader): Torch DataLoader object to load data
            master_bar (fastprogress.master_bar): Will be iterated over to draw
                batches and show validation progress

        Returns:
            float, float, torch.Tensor shape (10,10): Mean loss on validation set,
                fraction of correct predictions on validation set (accuracy)
        """

        num_classes = self.params["n_classes"] 

        epoch_loss = []
        epoch_y = []
        epoch_ypred = []
        epoch_correct, epoch_total = 0, 0
        confusion_matrix = torch.zeros(num_classes, num_classes)

        self.model.eval()
        with torch.no_grad():
            for x, y in fastprogress.progress_bar(dataloader, parent=master_bar):
                # make a prediction on validation set
                y_pred = self.model(x.to(self.device))

                # save the true and predicted labels
                epoch_y.append(y.tolist())
                epoch_ypred.append(y_pred.argmax(dim=1).cpu().tolist())

                # For calculating the accuracy, save the number of correctly
                # classified images and the total number
                epoch_correct += sum(y.to(self.device) == y_pred.argmax(dim=1))
                epoch_total += len(y)

                # Fill confusion matrix
                for (y_true, y_p) in zip(y, y_pred.argmax(dim=1)):
                    confusion_matrix[int(y_true), int(y_p)] +=1

                # Compute loss
                loss = self.criterion(y_pred, y.to(self.device))

                # For plotting the train loss, save it for each sample
                epoch_loss.append(loss.item())

            epoch_y = sum(epoch_y, [])
            epoch_ypred = sum(epoch_ypred, [])
            epoch_y = np.array(epoch_y).flatten()
            epoch_ypred = np.array(epoch_ypred).flatten()
            epoch_y = torch.tensor(epoch_y)
            epoch_ypred = torch.tensor(epoch_ypred)
            # compute f1score
            f1score = multiclass_f1_score(epoch_y, epoch_ypred, num_classes=self.params["n_classes"], average="macro")
            # compute recall
            recall = multiclass_recall(epoch_y, epoch_ypred, num_classes=self.params["n_classes"], average="macro")
            # compute precision
            precision = multiclass_precision(epoch_y, epoch_ypred, num_classes=self.params["n_classes"], average="macro")
            #gmean 
            gmean = weighted_geometric_mean_score(epoch_y, epoch_ypred)

        # Return the mean loss, the accuracy and the confusion matrix
        return np.mean(epoch_loss), accuracy(epoch_correct, epoch_total), confusion_matrix, float(f1score.cpu()), float(precision.cpu()), float(recall.cpu()), float(gmean)
    
    def run_training(self, train_dataloader, val_dataloader):
        """Run model training.

        Args:
            train_dataloader (DataLoader): Torch DataLoader object to load the
                training data
            val_dataloader (DataLoader): Torch DataLoader object to load the
                validation data

        Returns:
            list, list, list, list, torch.Tensor shape (10,10): Return list of train
                losses, validation losses, train accuracies, validation accuracies
                per epoch and the confusion matrix evaluated in the last epoch.
        """

        num_epochs = self.params["num_epochs"]
        verbose = self.params["verbose"]


        start_time = time.time()
        master_bar = fastprogress.master_bar(range(num_epochs))
        train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s = [],[],[],[],[],[]
        train_gmeans, val_gmeans = [],[]
        train_pres, val_pres, train_recs, val_recs = [],[],[],[]

        for epoch in master_bar:
            # Train the model
            train_loss, train_acc, train_f1, train_pre, train_rec, train_gmean = self.training_step(train_dataloader, master_bar)
            # Validate the model
            val_loss, val_acc, confusion_matrix, val_f1, val_pre, val_rec, val_gmean = self.validatation_step(val_dataloader, master_bar)
            
            # Save loss and acc for plotting
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            train_accs.append(train_acc)
            val_accs.append(val_acc)

            train_f1s.append(train_f1)
            val_f1s.append(val_f1)

            train_gmeans.append(train_gmean)
            val_gmeans.append(val_gmean)

            train_pres.append(train_pre)
            val_pres.append(val_pre)

            train_recs.append(train_rec)   
            val_recs.append(val_rec)

            # scheduler to adjust learning rate
            if self.scheduler: 
                self.scheduler.step(val_loss)
                # Get the learning rate from the optimizer
                lr = self.optimizer.param_groups[0]['lr']
                print("Learning Rate:", lr)

            if verbose:
                print('Train loss:', train_loss, 'val loss:', val_loss, 'train acc:', train_acc, 'val acc:', val_acc)
                # master_bar.write(f'Train loss: {epoch_train_loss:.2f}, val loss: {epoch_val_loss:.2f},train acc: {epoch_train_acc:.3f}, val acc: {epoch_val_acc:.3f}')

            # Early stopping to prevent overfitting
            if (epoch>10 and self.early_stopper):
                self.early_stopper.update(val_loss, self.model)
                if self.early_stopper.early_stop:
                    self.model = self.early_stopper.load_checkpoint(self.model)
                    break

        time_elapsed = np.round(time.time() - start_time, 0).astype(int)
        print(f'Finished training after {time_elapsed} seconds.')
        return train_losses, val_losses, train_accs, val_accs, confusion_matrix, train_f1s, val_f1s, train_gmeans, val_gmeans, train_pres, val_pres, train_recs, val_recs

    def get_confusion_matrix(self, test_loader, view_cm=False):
        num_classes = self.params["n_classes"] 

        confusion_matrix = torch.zeros(num_classes, num_classes)
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
        if view_cm:
            print(confusion_matrix, "\n")
        cm = confusion_matrix.numpy()
        return cm
    
    def testing_step(self, test_loader):
        "Function to get the test evaluation on the respective model"
        correct = 0
        f1score = 0
        total = 0
        self.model.eval()

        y, ypred = [], []

        with torch.no_grad():
            for data in test_loader:
                x, labels = data
                x, labels = x.to(self.device), labels.to(self.device)
                outputs = self.model(x)
                _, predicted = torch.max(outputs.data, 1)

                y.append(labels.cpu().tolist())
                ypred.append(predicted.cpu().tolist())

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                f1score += multiclass_f1_score(predicted, labels, num_classes=self.params["n_classes"], average="macro")

            epoch_y = sum(y, [])
            epoch_ypred = sum(ypred, [])
            epoch_y = np.array(epoch_y).flatten()
            epoch_ypred = np.array(epoch_ypred).flatten()
            y = torch.tensor(epoch_y)
            ypred = torch.tensor(epoch_ypred)
            # compute f1score
            f1score = multiclass_f1_score(y, ypred, num_classes=self.params["n_classes"], average="macro")
            # compute recall
            recall = multiclass_recall(y, ypred, num_classes=self.params["n_classes"], average="macro")
            # compute precision
            precision = multiclass_precision(y, ypred, num_classes=self.params["n_classes"], average="macro")
            #gmean 
            gmean = weighted_geometric_mean_score(y, ypred)
        
        return correct, total, accuracy(correct, total), float(f1score.cpu()), float(gmean), float(precision.cpu()), float(recall.cpu())
    

import torch

def normalize_data(x_train, x_test, x_val, norm_type):
    """
    Normalize the data based on the specified normalization type.
    data [sample, timesteps, channels]

    Parameters:
        x_train (torch.Tensor): Training data of shape [samples, time, channels].
        x_test (torch.Tensor): Test data of shape [samples, time, channels].
        x_val (torch.Tensor): Validation data of shape [samples, time, channels].
        norm_type (str): Type of normalization to apply.

    Returns:
        x_train_scaled (torch.Tensor): Normalized training data.
        x_test_scaled (torch.Tensor): Normalized test data.
        x_val_scaled (torch.Tensor): Normalized validation data.
    """
    if norm_type == "per-channel":
        # Compute mean and std for each channel across all samples and time steps
        train_mean = x_train.mean(dim=(0, 1))  # Shape: [channels]
        train_std = x_train.std(dim=(0, 1))    # Shape: [channels]

        # Avoid division by zero
        train_std[train_std == 0] = 1.0

        # Normalize the train, test, and validation datasets
        x_train_scaled = (x_train - train_mean) / train_std
        x_test_scaled = (x_test - train_mean) / train_std
        x_val_scaled = (x_val - train_mean) / train_std

    elif norm_type == "per-timestep":
        # Compute mean and std for each time step across all samples and channels
        train_mean = x_train.mean(dim=(0, 2), keepdim=True)  # Shape: [1, time, 1]
        train_std = x_train.std(dim=(0, 2), keepdim=True)    # Shape: [1, time, 1]

        # Avoid division by zero
        train_std[train_std == 0] = 1.0

        # Normalize the train, test, and validation datasets
        x_train_scaled = (x_train - train_mean) / train_std
        x_test_scaled = (x_test - train_mean) / train_std
        x_val_scaled = (x_val - train_mean) / train_std

    else:
        raise ValueError(f"Unsupported normalization type: {norm_type}")

    print("shapes:", x_train_scaled.shape, x_test_scaled.shape, x_val_scaled.shape)
    return x_train_scaled, x_val_scaled, x_test_scaled

    
def data_to_tensor(data, labels):

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels)

    torch_lbls = torch.tensor(labels)
    torch_data = torch.tensor(data) # this type is needed for mps devices as float64 is not supported 
    torch_tensor = torch.utils.data.TensorDataset(torch_data, torch_lbls) 

    return torch_tensor

def accuracy(correct, total):
    """Compute accuracy as percentage.

    Args:
        correct (int): Number of samples correctly predicted.
        total (int): Total number of samples

    Returns:
        float: Accuracy
    """
    return float(correct)/total

def plot_progress(title, label, train_results, val_results, yscale='linear', save_path=None,
         extra_pt=None, extra_pt_label=None):
    """Plot learning curves.

    Args:
        title (str): Title of plot
        label (str): x-axis label
        train_results (list): Results vector of training of length of number
            of epochs trained. Could be loss or accuracy.
        val_results (list): Results vector of validation of length of number
            of epochs. Could be loss or accuracy.
        yscale (str, optional): Matplotlib.pyplot.yscale parameter.
            Defaults to 'linear'.
        save_path (str, optional): If passed, figure will be saved at this path.
            Defaults to None.
        extra_pt (tuple, optional): Tuple of length 2, defining x and y coordinate
            of where an additional black dot will be plotted. Defaults to None.
        extra_pt_label (str, optional): Legend label of extra point. Defaults to None.
    """
    epoch_array = np.arange(len(train_results)) + 1
    train_label, val_label = "Training "+label.lower(), "Validation "+label.lower()

    sns.set(style='ticks')

    fig, ax = plt.subplots(figsize=(4, 6))
    ax.plot(epoch_array, train_results, epoch_array, val_results, linestyle='dashed', marker='o')
    legend = ['Train results', 'Validation results']

    if extra_pt:
        x, y = extra_pt
        ax.scatter(x, y,s=120, c='black', marker='o', label=extra_pt_label)
        legend.append(extra_pt_label)

    ax.legend(legend)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(label)
    ax.set_yscale(yscale)
    ax.set_title(title)

    sns.despine(trim=True, offset=5)
    ax.set_title(title, fontsize=15)
    if save_path:
        fig.savefig(str(save_path), bbox_inches='tight')
    plt.show()



def get_metrics_from_confusion_matrix(cm, chosen_index):
    total = np.sum(cm)
    idx = chosen_index
    recall = cm[idx][idx] / np.sum(cm[:, idx])
    precision = cm[idx][idx] / np.sum(cm[idx])
    accuracy = cm[idx][idx] + (total - np.sum(cm[idx]) - np.sum(cm[:, idx]) + cm[idx][idx])
    return recall, precision, accuracy / total


def check_precision_recall_accuracy(cm, all_classes):
    "Function to getting the "
    for i, _class in enumerate(all_classes):
        recall, precision, accuracy = get_metrics_from_confusion_matrix(cm, i)
        print(f"{_class} - recall : ", recall, " precision : ", precision, " accuracy : ", accuracy)


def visualize_confusion_matrix(cm, classes, correct, total, path=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm / (cm.astype("float").sum(axis=1) + 1e-9), annot=True, ax=ax)

    # labels, title and ticks for plotting the confusion matrix
    ax.set_xlabel('Predicted', size=18)
    ax.set_ylabel('True', size=18)
    ax.set_title('Confusion Matrix', size=18)
    ax.xaxis.set_ticklabels(classes, size=13)
    ax.yaxis.set_ticklabels(classes, size=13)
    if path: ax.figure.savefig(path)
    print("correctness:", correct / total)

def weighted_geometric_mean_score(y_true, y_pred):
    """
    Compute the weighted geometric mean score for multi-class classification.

    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        float: Weighted geometric mean score.
    """
    # Get the unique classes
    classes = np.unique(y_true)
    num_classes = len(classes)
    
    # Initialize the confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    # Populate the confusion matrix
    for i in range(len(y_true)):
        true_class = y_true[i]
        pred_class = y_pred[i]
        cm[true_class, pred_class] += 1
    
    # Initialize variables to store sensitivity and specificity
    sensitivity = []
    specificity = []
    
    # Iterate over each class
    for i in range(num_classes):
        # True Positives (TP): Diagonal element for the current class
        TP = cm[i, i]
        
        # False Negatives (FN): Sum of row i excluding TP
        FN = np.sum(cm[i, :]) - TP
        
        # False Positives (FP): Sum of column i excluding TP
        FP = np.sum(cm[:, i]) - TP
        
        # True Negatives (TN): Sum of all elements except row i and column i
        TN = np.sum(cm) - (TP + FP + FN)
        
        # Sensitivity (Recall) for the current class
        sensitivity_i = TP / (TP + FN) if (TP + FN) != 0 else 0
        
        # Specificity for the current class
        specificity_i = TN / (TN + FP) if (TN + FP) != 0 else 0
        
        sensitivity.append(sensitivity_i)
        specificity.append(specificity_i)
    
    # Convert lists to numpy arrays
    sensitivity = np.array(sensitivity)
    specificity = np.array(specificity)
    
    # Compute the geometric mean for each class
    geometric_mean = np.mean(np.sqrt(sensitivity * specificity))
    
    return geometric_mean