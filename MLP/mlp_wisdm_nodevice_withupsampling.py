
import os
import sys
import random
random.seed(43)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch

from imblearn.over_sampling import SMOTE, ADASYN

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger().setLevel(logging.ERROR)

from sklearn.utils import class_weight

import MLP.run_training_testing_mpl as run_traintest_mlp


def plot_samples(num_samples, dataf):
    fig, ax =plt.subplots(num_samples,len(class_labels), figsize=(8, 3.5))

    for jth,usr in enumerate(random.sample(sorted(num_usrs), num_samples)):
        for idx, act in enumerate(class_labels):
            if jth == 0:ax[jth,idx].set_title(act)
            ax[jth,idx].set_ylim(min(dataf['x-accel'].min(), dataf['y-accel'].min(), dataf['z-accel'].min()),
                                max(dataf['x-accel'].max(), dataf['y-accel'].max(), dataf['z-accel'].max()))
            # ax[jth,idx].set_xlim(dataf['timestamp'].min(), dataf['timestamp'].max())
            ax[jth,idx].set_yticklabels([])
            ax[jth,idx].set_xticklabels([])
            ax[jth,idx].sharey(ax[jth,0])
            if idx==0: ax[jth,idx].set_ylabel(f'usr-{usr}')
            tmpdf = dataf.loc[(dataf['user'] == usr) & (dataf['activity'] == act)]
            ax[jth,idx].scatter(tmpdf['timestamp'], tmpdf['x-accel'], label='x-accel', s=0.2)
            ax[jth,idx].scatter(tmpdf['timestamp'], tmpdf['y-accel'], label='y-accel', s=0.2)
            ax[jth,idx].scatter(tmpdf['timestamp'], tmpdf['z-accel'], label='z-accel', s=0.2)
    plt.tight_layout(pad=0., w_pad=0., h_pad=0)
    plt.show()

try:

    if len(sys.argv) < 7:
        print("add path to data (with slash at the end of path), number of hidden layers, hidden layer size, batch size, patience, model, lr, upsample and pathsave")
    else:
        pathdata = sys.argv[1]
        num_hidden_lyr = int(sys.argv[2])
        hidden_lyr_size = int(sys.argv[3])
        batch_size = int(sys.argv[4])
        patience = int(sys.argv[5])
        mlpmodel = sys.argv[6]
        lr = float(sys.argv[7])
        upsample = sys.argv[8]
        pathsave = sys.argv[9]
    print("num_hidden_lyr, hidden_lyr_size, batch_size, patience, mlpmodel, lr, upsample:", num_hidden_lyr, hidden_lyr_size, batch_size, patience, mlpmodel, lr, upsample)

    # find device
    if torch.cuda.is_available(): # NVIDIA
        device = torch.device('cuda')
    elif torch.backends.mps.is_available(): # apple silicon
        device = torch.device('mps') 
    else:
        device = torch.device('cpu') # fallback
    device

    num_cpus = os.system("taskset -c -p 0-95 %d" % os.getpid()) #os.cpu_count()
    print(num_cpus, 'CPUs available')


    datafldr =pathdata
    dataai="data/WISDM_ar_v1.1/"
    datapath = datafldr + dataai
    col_names = ['user', 'activity', 'timestamp', 'x-accel', 'y-accel', 'z-accel']

    df = pd.read_csv(datapath+"WISDM_ar_v1.1_raw.txt",
                    header=None, names=col_names, delimiter=',', comment=';',
                        on_bad_lines='skip') #skip/warn bad lines
    print(df.shape)

    #number of users
    num_usrs = df['user'].unique()
    print(num_usrs.shape)

    #number of activities
    class_labels = df.activity.unique()
    num_classes = len(class_labels)
    print(class_labels)

    # encoding the activity into integers
    def encodedf(df):
        le = LabelEncoder()
        df['activity_encoded'] = le.fit_transform(df['activity'])
        return df
    df = encodedf(df.dropna())
    print(df.head())
    print(df.groupby(['activity_encoded', 'activity']).size())

    # separate data into features and target
    x = df.drop(columns=['activity', 'activity_encoded', 'user', 'timestamp'])
    y = df['activity_encoded']
    num_features = x.shape[1]
    print(x.shape, num_features)
    print(x.head())
    print(y.head())

    # upsampling
    print("upsampling:", upsample)
    if upsample == "SMOTE":
        upsampling = SMOTE    
    else:
        upsampling = ADASYN
    x, y = upsampling().fit_resample(x, y)

    # Computed class weights
    ynumpy = y.to_numpy()
    class_weights=class_weight.compute_class_weight(class_weight="balanced",
                                                     classes=np.unique(ynumpy), y=ynumpy) 
    class_weights=torch.tensor(class_weights,dtype=torch.float)

    print(np.unique(ynumpy),class_weights)
    print(class_weights.sum(axis=0))

    # running the model
    print("running the model")
    run_traintest_mlp.run_mlp_training(x,y,class_labels, device, num_features, num_classes,
                                       test_size=0.2, val_size=0.2, batch_size=batch_size,
                                         num_cpus=num_cpus,lr= lr, num_epochs=1000,
                                         patience=patience, modeltype=str(mlpmodel), 
                                         num_hidden_lyr=num_hidden_lyr, 
                                         hidden_lyr_size=hidden_lyr_size, 
                                         verbose=True, pathsave=str(pathsave), 
                                         weights=class_weights)

except Exception as e:
    print("An exception occurred:", str(e))
