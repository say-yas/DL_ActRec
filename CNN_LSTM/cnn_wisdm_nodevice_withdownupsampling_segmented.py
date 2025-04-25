
import os
import sys
import random
random.seed(43)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch

from imblearn.combine import SMOTEENN,SMOTETomek

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger().setLevel(logging.ERROR)

from sklearn.utils import class_weight

# import CNN_LSTM.run_training_testing_cnnlstm as run_traintest_cnnlstm
import CNN_LSTM.run_training_testing_cnnlstm_multisplit as run_traintest_cnnlstm


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

    if len(sys.argv) < 14:
        print("path_data/, max_length_series, num_conv_lyrs, size_linear_lyr, num_blocks_per_layer, initial_channels, lstm_hidden_size, num_lstm_layers, batch_size, patience, cnnlstmmodel, norm_type, lr, segment_duration")
    else:
        pathdata = sys.argv[1]
        max_length_series = int(sys.argv[2])
        num_conv_lyrs = int(sys.argv[3])
        size_linear_lyr = int(sys.argv[4])
        num_blocks_per_layer = int(sys.argv[5])
        initial_channels = int(sys.argv[6])
        lstm_hidden_size = int(sys.argv[7])
        num_lstm_layers = int(sys.argv[8])
        batch_size = int(sys.argv[9])
        patience = int(sys.argv[10])
        cnnlstmmodel = sys.argv[11]
        norm_type = sys.argv[12]
        lr = float(sys.argv[13])
        segment_duration = int(sys.argv[14])
        doupsample = sys.argv[15]
        pathsave = sys.argv[16]
        if len(sys.argv) >= 17:
            num_training = int(sys.argv[17])
        else:
            num_training = 1
    
    print("max_length_series:", max_length_series)
    print("num_conv_lyrs:", num_conv_lyrs)
    print("size_linear_lyr:", size_linear_lyr)
    print("num_blocks_per_layer:", num_blocks_per_layer)
    print("initial_channels:", initial_channels)
    print("lstm_hidden_size:", lstm_hidden_size)
    print("num_lstm_layers:", num_lstm_layers)
    print("batch_size:", batch_size)
    print("patience:", patience)
    print("cnnlstmmodel:", cnnlstmmodel)
    print("norm_type:", norm_type)
    print("lr:", lr)
    print("segment_duration:", segment_duration)
    print("doupsample:", doupsample)
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

    # add segments
    print("introducing segments")
    segment_duration = segment_duration
    # Calculate the number of segments per sample
    num_segments = padded_series.shape[1] // segment_duration
    print(" shape of all segmented data", padded_series.shape[0]*num_segments, segment_duration, padded_series.shape[2])
    reshaped_data = padded_series.reshape(padded_series.shape[0], num_segments, segment_duration, padded_series.shape[2])

    valid_segments = []
    valid_labels = []

    # Iterate over each sample and its segments to remove segments with all zeros x,y,z
    for sample_idx in range(reshaped_data.shape[0]):
        for segment_idx in range(reshaped_data.shape[1]):
            segment = reshaped_data[sample_idx, segment_idx]
            if not (torch.all(segment[:, 0]==0) and torch.all(segment[:, 1]==0) and torch.all(segment[:, 2]==0) ):
                valid_segments.append(segment)
                valid_labels.append(labels[sample_idx])

    padded_series = torch.stack(valid_segments)
    labels = torch.tensor(valid_labels)
    max_length = segment_duration
    print(" shape of segmented data with nonzero features",padded_series.shape, labels.shape)

    # downupsampling
    print("downupsampling:", doupsample)
    num_sample, timesample, channels = padded_series.shape
    padded_series = padded_series.reshape(num_sample, timesample*channels)
    if doupsample == "SMOTEENN":
        doupsampling = SMOTEENN(random_state=43)   
    else:
        doupsampling = SMOTETomek(random_state=43)   
    padded_series, labels = doupsampling.fit_resample(padded_series, labels)
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
    run_traintest_cnnlstm.run_cnnlstm_training(padded_series, labels, class_labels,device, 
                                               num_channels, num_classes,test_size=0.2,val_size=0.2, 
                                               batch_size=batch_size, num_cpus=num_cpus,lr=lr, num_epochs=50000, 
                                               patience=patience, modeltype = cnnlstmmodel, max_length_series=max_length, 
                                               num_conv_layers=num_conv_lyrs, size_linear_lyr=size_linear_lyr, 
                                               num_blocks_per_layer=num_blocks_per_layer, initial_channels=initial_channels,
                                               lstm_hidden_size=lstm_hidden_size, num_lstm_layers=num_lstm_layers, opt="adamW", 
                                               verbose=True, pathsave=pathsave, weights=class_weights, norm_type=norm_type, num_training=num_training)

except Exception as e:
    print("An exception occurred:", str(e))
