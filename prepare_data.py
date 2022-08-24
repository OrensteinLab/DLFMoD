# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 15:53:50 2020

@author: yahelsalomon
"""

import os
from Bio import SeqIO
from numpy.random import shuffle
from keras.utils import to_categorical
import numpy as np
import pandas as pd

def DataPreparation(root_dir, pos_filedir, neg_filedir, df_pos_train_filedir,
                    df_train_filedir, df_pos_test_filedir, df_test_filedir,
                    params):
    '''
    Arange train and test dataset stractures.
    
    Args:
        root_dir             : Path of the main directory.
        pos_filedir          : Path to the directory of the positives fasta file.
        neg_filedir          : Path to the directory of the negatives fasta file. 
        df_pos_train_filedir : Path to the directory which will store the positives training dataset as csv file.
        df_train_filedir     : Path to the directory which will store the training dataset as csv file.
        df_pos_test_filedir  : Path to the directory which will store the positives test dataset as csv file.
        df_test_filedir      : Path to the directory which will store the test dataset as csv file.
        params               : Hyper-parameters dictionary.
        
    Returns:
        x_train : Training set as 4xL one-hot encoded matrices.
        y_train : Training labels as 1xL vector of 1's and 0's.
        x_test  : Test set as 4xL one-hot encoded matrices.
        y_test  : Test lables as 1xL vector of 1's and 0's.
        
    '''
    train_data = []; test_data = []
    # 1-label
    train_data,test_data = LoadData(train_data,test_data,pos_filedir,1)
    # save positives dataset
    x_train, y_train = ArrangeData(train_data)
    x_test, y_test = ArrangeData(test_data)
    SaveDataset(x_train, y_train, df_pos_train_filedir)
    SaveDataset(x_test, y_test, df_train_filedir)
    # 0-label
    train_data,test_data = LoadData(train_data,test_data,neg_filedir,0)                          
    x_train, y_train = ArrangeData(train_data)
    x_test, y_test = ArrangeData(test_data)
    # save dataset
    SaveDataset(x_train, y_train, df_pos_test_filedir)
    SaveDataset(x_test, y_test, df_test_filedir)

    x_train, y_train = DataEncoding(x_train, y_train, params['num_classes'])
    x_test, y_test = DataEncoding(x_test, y_test, params['num_classes'])
    
    return x_train, y_train, x_test, y_test


    
def LoadData(train_data,test_data,filedir,label):
    '''
    Load sequences and add them to the train and test structures.
    
    Args:
        train_data  : Training dataset list.
        test_data   : Test dataset list.
        filedir     : Path to the Path to the directory of the fasta file.
        label       : Positives or negatives (1/0).
        
    Returns:
        train_data  : Training dataset list.
        test_data   : Test dataset list.   
    '''
    for filename in os.listdir(filedir):
        if filename.endswith(".fa"):
            list_of_records = list(SeqIO.parse(filedir + filename, "fasta"))
            for record in list_of_records:
                record.seq = record.seq.upper()
                if record.id.split(':')[0]!='chr1':
                    train_data += [str(record.seq),label]
                else:
                    test_data += [str(record.seq),label]
    return train_data,test_data


def ArrangeData(data):
    '''
    Reshape the data, and return it as two sequences and lables numpy arrays.
    
    Args:
        data        : Sequences and lables dataset.
        
    Returns:
        data[:,0]   : Sequences.
        data[:,1]   : Lables.
    '''
    data = [data[x:x+2] for x in range(0, len(data),2)] # reshape the dataset
    data = np.array(data)
    shuffle(data)
    return data[:,0], data[:,1]
                    
def SaveDataset(x_data, y_data, file_name):
    '''
    Reshape the data, and return it as two sequences and lables numpy arrays.
    
    Args:
        x_data      : Sequences.
        y_data      : Lables.
        file_name   : Path to the directory which will store the dataset as csv file.
    '''
    raw_data = {'seq': x_data,
                'label': y_data}
    df = pd.DataFrame(raw_data, columns = ['seq', 'label'])
    df.to_csv(file_name)
    

def DataEncoding(x_data, y_data, num_classes):
    '''
    Every sequence is encoded as a one-hot matrix.
    Lables vector is converted to categorical format.
    
    Args: 
        x_data      : Sequences.
        y_data      : Lables.
        
    Returns:
        x_data      : Dataset as 4xL one-hot encoded matrices.
        y_data      : Data lables as 1x2 categorical encoded vectores.
    '''
    x_data = np.array([OneHot(sample) for sample in x_data])
    y_data = np.array([float(label) for label in y_data])
    #y_data = to_categorical(y_data, num_classes)
    return x_data,y_data

def OneHot(string):
    '''
    Every nucleotide is encoded as a one-hot vector.
    Args: 
        string      : String over ACGTN.
        
    Returns:
        oneHot_arr  : String represented as 4xL one-hot encoded matrix.
    
    '''
    trantab=str.maketrans('ACGTN','01234')
    string=string+'ACGTN'
    data=list(string.translate(trantab))
    oneHot_arr = to_categorical(data)[0:-5]
    oneHot_arr = oneHot_arr[:,:-1]
    return oneHot_arr