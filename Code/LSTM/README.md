**Steps to run the LSTM model:**

_Structure of the file_

main.py ---> This is the main project file which contains data preprocessing,pretrained layer,compiling model,predicting intents and responses and chat bot UI integration 

utilities.py ---> This file contains all the utilities required for the main.py (read_glove_vecs,softmax,read_csv,convert_to_one_hot...)

glove.6B.50d.txt, glove.6B.100d.txt ---> Gloves text files Pre-trained word vectors

trained_lstm_128_128_dropout_4_3.h5 ---> pretrained lstm model

After executing main.py we generate the following files,

tags.txt ---> storing all the tags in a txt file

trained_lstm.h5 ---> saved model

Gloves text files , pretrained model, saved model can be found here(since files are greater than 25mb)
https://drive.google.com/drive/folders/1R6co2g-IDEobIf8z_DOc7yuUTDETB_xH?usp=sharing



