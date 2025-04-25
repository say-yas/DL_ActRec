
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
#import Transformer.run_training_testing_transformer as run_traintest_transformer
import Transformer.run_training_testing_transformer_multisplit as run_traintest_transformer


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

    if len(sys.argv) < 16: 
        print("path_data/, max_length_series, embed_size, nhead_encoder, dim_feedforward, num_encoderlayers, conv1d_kernel_size, size_linear_layers, batch_size, patience, transformermodel, norm_type, lr")
    else:
        pathdata = sys.argv[1]
        max_length_series = int(sys.argv[2])
        transformermodel = sys.argv[3]
        embed_size = int(sys.argv[4])
        nhead_encoder = int(sys.argv[5])
        dim_feedforward = int(sys.argv[6])
        num_encoderlayers = int(sys.argv[7])
        conv1d_kernel_size = int(sys.argv[8])
        size_linear_layers = int(sys.argv[9])
        bool_conv1d_emb = sys.argv[10]
        dropout = float(sys.argv[11])
        batch_size = int(sys.argv[12])
        patience = int(sys.argv[13])
        norm_type = sys.argv[14]
        lr = float(sys.argv[15])
        upsample = sys.argv[16]
        pathsave = sys.argv[17]
        if len(sys.argv) >= 18:
            num_training = int(sys.argv[18])
        else:
            num_training = 1
    
    print("max_length_series:", max_length_series)
    print("batch_size:", batch_size)
    print("patience:", patience)
    print("norm_type:", norm_type)
    print("transformermodel:", transformermodel)
    print("embed_size:", embed_size)
    print("nhead_encoder:", nhead_encoder)
    print("dim_feedforward:", dim_feedforward)
    print("num_encoderlayers:", num_encoderlayers)
    print("conv1d_kernel_size:", conv1d_kernel_size)
    print("bool_conv1d_emb:", bool_conv1d_emb)
    print("dropout:", dropout)
    print("size_linear_layers:", size_linear_layers)
    print("lr:", lr)
    print("upsample:", upsample)
    print("num_training:", num_training)

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
    num_channels = 3 # x-accel, y-accel, z-accel
    num_classes = len(class_labels)
    print(class_labels)

    activity_to_label = {activity: i for i, activity in enumerate(df['activity'].unique())}

    # Group by users and activity
    grouped = df.groupby(['user', 'activity'])

    # Combine X, Y, Z into a single time-series for each group
    time_series = []
    labels = []
    activity_to_label = {activity: i for i, activity in enumerate(df['activity'].unique())}

    for (user, activity), group in grouped:
        # Stack X, Y, Z into a single array of shape (timesteps, 3)
        series = np.column_stack((group['x-accel'], group['y-accel'], group['z-accel']))
        time_series.append(series)
        labels.append(activity_to_label[activity])

    print(time_series[0].shape)
    print(len(labels), np.unique(labels, return_counts=True))

    # Pad/truncate to a fixed length
    max_length = max_length_series  # Choose a fixed length
    padded_series = torch.nn.utils.rnn.pad_sequence([torch.tensor(series, dtype=torch.float32) for series in time_series],
                                batch_first=True, padding_value=0)
    print(padded_series.shape)
    padded_series = padded_series[:, :max_length, :]  # Truncate to max_length if necessary
    print(padded_series.shape)

    # upsampling
    num_sample, timesample, channels = padded_series.shape
    padded_series = padded_series.reshape(num_sample, timesample*channels)
    print("upsampling:", upsample)
    if upsample == "SMOTE":
        upsampling = SMOTE(sampling_strategy = 'minority', random_state=43)    
    else:
        upsampling = ADASYN(sampling_strategy = 'minority', random_state=43)
    padded_series, labels = upsampling.fit_resample(padded_series, labels)
    num_sample, timesample = padded_series.shape
    padded_series = padded_series.reshape(num_sample, timesample//channels, channels)
    padded_series = torch.tensor(padded_series, dtype=torch.float32)
    print(padded_series.shape)

    # Computed class weights
    ynumpy = np.array(labels)
    class_weights=class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(ynumpy), y=ynumpy) 
    class_weights=torch.tensor(class_weights,dtype=torch.float)

    print(np.unique(ynumpy),class_weights)
    print("class weights:", class_weights.sum(axis=0))

    # running the model
    print("running the model")
    run_traintest_transformer.run_transformer_training(padded_series, labels, class_labels,device, 
                                               num_channels, num_classes,test_size=0.2,val_size=0.2, 
                                               batch_size=batch_size, num_cpus=num_cpus,lr=lr, num_epochs=5000, 
                                               patience=patience, modeltype = transformermodel, max_length_series=max_length, 
                                               embed_size=embed_size, nhead=nhead_encoder, 
                                               dim_feedforward=dim_feedforward, num_encoderlayers=num_encoderlayers,
                                               dropout = dropout, conv1d_emb = bool_conv1d_emb,
                                               conv1d_kernel_size=conv1d_kernel_size, size_linear_layers=size_linear_layers, opt="adamW", 
                                               verbose=True, pathsave=pathsave, weights=class_weights, norm_type=norm_type, num_training=num_training)

except Exception as e:
    print("An exception occurred:", str(e))
