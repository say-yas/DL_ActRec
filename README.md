# Deep learning for activity recognition
This package is implemented by <ins>Sharareh Sayyad</ins>. It explores the classification problem for a time-dependent dataset. In particular, the code is implemented to address this problem on [WISDM dataset](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset).



## Model architectures

Various deep learning models are implemented in this package, including:
- MLP: The codes are stored in [MLP/](https://github.com/say-yas/DL_ActRec/tree/main/MLP) 
- CNN: The codes are stored in [CNN_LSTM/](https://github.com/say-yas/DL_ActRec/tree/main/CNN_LSTM) 
- LSTM: The codes are stored in [CNN_LSTM/](https://github.com/say-yas/DL_ActRec/tree/main/CNN_LSTM) 
- CNN-LSTM: The codes are stored in [CNN_LSTM/](https://github.com/say-yas/DL_ActRec/tree/main/CNN_LSTM) 
- Transformer: The codes are stored in [Transformer/](https://github.com/say-yas/DL_ActRec/tree/main/Transformer) 

A notebook to analyze the statistical behavior of the dataset is available [here](https://github.com/say-yas/DL_ActRec/tree/main/data_analysis).

## Requirements
A collection of required packages to install and execute our code is collected in `requirements.txt`.

## Installation
 To install the package, one can either call `sh install.sh` or try `pip install .` inside the package's main directory.

## How to use
After installing the package, all provided notebooks and Python codes in `data_analysis/MLP/CNN_LSTM/Transformer` can be called.

See the [report file](https://github.com/say-yas/DL_ActRec/tree/main/Report_Activity_Recognition.pdf) for further details.

## Summary of best results
I have collected a selected number of my best performed parameter sets and their associated metrics on the test sets in [postprocessing_avg/](https://github.com/say-yas/DL_ActRec/tree/main/postprocessing_avg). The results are obtained from averaging the outputs of each particular model for 20 different trainings.
