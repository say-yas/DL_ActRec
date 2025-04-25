import os
import sys
import random
random.seed(43)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch

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
            # ax[jth,idx].set_ylim(min(dataf['x-accel'].min(), dataf['y-accel'].min(), dataf['z-accel'].min()),
            #                     max(dataf['x-accel'].max(), dataf['y-accel'].max(), dataf['z-accel'].max()))
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

try :

    if len(sys.argv) < 7:
        print("add path to data (with slash at the end of path), number of hidden layers, hidden layer size, batch size, patience, model, lr and pathsave")
    else:
        pathdata = sys.argv[1]
        num_hidden_lyr = int(sys.argv[2])
        hidden_lyr_size = int(sys.argv[3])
        batch_size = int(sys.argv[4])
        patience = int(sys.argv[5])
        mlpmodel = sys.argv[6]
        lr = float(sys.argv[7])
        pathsave = sys.argv[8]
    print("num_hidden_lyr, hidden_lyr_size, batch_size, patience, mlpmodel, lr:", num_hidden_lyr, hidden_lyr_size, batch_size, patience, mlpmodel, lr)

    # find device
    if torch.cuda.is_available(): # NVIDIA
        device = torch.device('cuda')
    elif torch.backends.mps.is_available(): # apple silicon
        device = torch.device('mps') 
    else:
        device = torch.device('cpu') # fallback
    device

    # number of cpus
    num_cpus = os.system("taskset -c -p 0-95 %d" % os.getpid()) # os.cpu_count()
    print(num_cpus, 'CPUs available')
 
    # ## Reading dataset
    df=pd.DataFrame(columns=['user', 'activity', 'timestamp', 'x-accel', 'y-accel', 'z-accel', 'device_type',
                            'sensor_type'])

    datafldr =pathdata
    dataai="data/"
    datapath = datafldr + dataai
    root_folder = "wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/wisdm-dataset/raw"
    names=['user', 'activity', 'timestamp', 'x-accel', 'y-accel', 'z-accel']

    for subfolder in ["phone", "watch"]:
        for sub_subfolder in ["gyro", "accel"]:
            folder_path = os.path.join(datapath+root_folder, subfolder, sub_subfolder)

            # Iterate through all files in the sub-sub-folder
            for file in os.listdir(folder_path):
                if file != '.DS_Store':
                    file_path = os.path.join(folder_path, file)

                    # Read the dataset (assuming CSV format, change if needed)
                    temp_df = pd.read_csv(file_path, names=names,  delimiter=',', comment=';',
                            on_bad_lines='skip')

                    # Add new features
                    temp_df["device_type"] = subfolder  # 'phone' or 'watch'
                    temp_df["sensor_type"] = sub_subfolder  # 'gyro' or 'accel'

                    # Append to list
                    df = pd.concat([df.astype(temp_df.dtypes), temp_df.astype(df.dtypes)], ignore_index=True) 
    print(df.shape)
    df.head()

    # number of users
    num_usrs = df['user'].unique()
    print(num_usrs.shape)

    # number of activities
    class_labels = df.activity.unique()
    num_classes = len(class_labels)
    print(class_labels)

    # encoding the activity into integers
    def encodedf(df):
        le = LabelEncoder()
        df['activity_encoded'] = le.fit_transform(df['activity'])
        df['sensor_type_encoded'] = le.fit_transform(df['sensor_type'])
        df['device_type_encoded'] = le.fit_transform(df['device_type'])
        return df
    df = encodedf(df)
    # print(df.head())
    df.drop(columns=['sensor_type', 'device_type'], inplace=True)
    print(df.head())
    print(df.groupby(['activity_encoded', 'activity']).size())

    # separate data into features and target
    x = df.drop(columns=['activity', 'activity_encoded', 'user', 'timestamp'])
    y = df['activity_encoded']
    num_features = x.shape[1]
    # print(x.shape, num_features)
    
    print(y.head())
    print(x.head())

    # Computed class weights
    ynumpy = y.to_numpy()
    class_weights=class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(ynumpy), y=ynumpy) 
    class_weights=torch.tensor(class_weights,dtype=torch.float)

    print(np.unique(ynumpy),class_weights)
    print(class_weights.sum(axis=0))


    # ## Training and Testing MLP model
    run_traintest_mlp.run_mlp_training(x,y,class_labels, device, num_features, num_classes,
                                       test_size=0.2, val_size=0.2, batch_size=batch_size,
                                       num_cpus=num_cpus, lr=lr, num_epochs=1000,  patience=patience,
                                       modeltype=str(mlpmodel), num_hidden_lyr=num_hidden_lyr,
                                       hidden_lyr_size=hidden_lyr_size, verbose=True,
                                       pathsave=str(pathsave), weights=class_weights)

except Exception as e:
    print("An exception occurred:", str(e))