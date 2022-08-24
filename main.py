#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yahelsalomon
"""

print(__doc__)

from IPython import get_ipython
import matplotlib
matplotlib.use('Agg')
import sys
import os
import numpy as np

from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from prepare_data import DataPreparation
from train import CrossValidationTraining, StraightforwardTraining, PlotStatistics

root_dir = os.getcwd()
os.listdir(root_dir)

# Read user input arguments
try:
    args = [x for x in sys.argv[1:]]
    params = {'mission':int(args[0]),
                  'task':int(args[1]),
                  'train_method':args[2],
                  'name':args[3],
                  'data':args[4],
                  'pool':int(args[5]),
                  'filters':int(args[6]),
                  'filter_size':int(args[7]),
                  'batch':int(args[8]),
                  'opt':args[9],
                  'epoch':int(args[10]),
                  }
    params['model_num'] = int(args[11])
except:
    print('You are tring to insert invalid input.')
params['num_classes']=2

space = {
        'drop': hp.choice( 'DROPOUT', ( 0.1,0.5)),
        'delta': hp.choice( 'DELTA', ( 1e-04,1e-06,1e-08)),
        'moment': hp.choice( 'MOMENT', (0.9, 0.99, 0.999 )),
        }

params.update(sample(space))


# ~~~~~~~~~~~~~~~~~~~~~~       Data Preprocessing       ~~~~~~~~~~~~~~~~~~~~~~~
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("1. Data preprocessing: generate positives and negatives structures")

pos_filedir = os.path.join(root_dir,'data','positives',str(params['data']))
neg_filedir = os.path.join(root_dir,'data','negatives',str(params['data']))
df_pos_train_filedir = os.path.join(root_dir,'data',str(params['data'])+'_Dataset_train_pos.csv')
df_train_filedir = os.path.join(root_dir,'data',str(params['data'])+'_Dataset_train.csv')
df_pos_test_filedir = os.path.join(root_dir,'data',str(params['data'])+'_Dataset_test_pos.csv')
df_test_filedir = os.path.join(root_dir,'data',str(params['data'])+'_Dataset_test.csv')

try:
    os.path.exists(os.path.dirname(pos_filedir))
except:
    os.makedirs(os.path.dirname(pos_filedir))
    print('Put the Positives fasta file in: '+os.path.dirname(pos_filedir))
try:
    os.path.exists(os.path.dirname(neg_filedir))
except:
    os.makedirs(os.path.dirname(neg_filedir))
    print('Put the Negatives fasta file in: '+os.path.dirname(pos_filedir))

x_train, y_train, x_test, y_test = DataPreparation(root_dir,pos_filedir, neg_filedir,
                                                   df_pos_train_filedir, df_train_filedir,
                                                   df_pos_test_filedir, df_test_filedir,
                                                   params)

# ~~~~~~~~~~~~~~       Evaluating Predicting Accuracy Task       ~~~~~~~~~~~~~~
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
assert params['task'] == "accuracy" or params['task'] == "motif", "There is no task named " + params['task']
assert params['train'] == "cross-validation" or params['train'] == "straightforward", "There is no task named " + params['train']

if params['task']=='accuracy': 
    
    if params['cross-validation']:
        # Hyper-parameters search using 10-fold cross validation 
        # the train set is all chromosomes except chromosome 1
        model, tprs, aucs, auprs, mean_fpr = CrossValidationTraining(root_dir,x_train,y_train,params)
    elif params['straightforward']:
        # Train the model with the best hyper-parameters
        # the train set is all chromosomes except chromosome 1
        model, tprs, aucs, auprs, mean_fpr = StraightforwardTraining(root_dir,x_train,y_train,x_test,y_test,params)
    
    # Plot information
    roc_filedir = os.path.join(root_dir,'images',params['neg'] + "_" + params['name'] +
                                 "_" + "AUC_" + str(params['model_num']) + "_day" + str(params['day']) +
                                 "_filters" + str(params['filters']) + "_filter_size" +
                                 str(params['filter_size']) +  "_batch" + str(params['batch']) +  "_" +
                                 params['opt'] + "_" + str(params['epoch']) + "epochs" + ".png")    
    PlotStatistics(roc_filedir,tprs, aucs, auprs, mean_fpr)


# ~~~~~~~~~~~~~~~~~~~~~       Motif Extraction Task       ~~~~~~~~~~~~~~~~~~~~~
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
elif params['task']=='motif': 
    
    x_data = np.concatenate((x_train,x_test))
    y_data = np.concatenate((y_train,y_test))
    
    model, tprs, aucs, auprs, mean_fpr = CrossValidationTraining(root_dir,x_train,y_train,params)
    
    # save model
    json_filedir = os.path.join(root_dir,"models",str(params['data'])+"_model_" + str(params['model_num']) + ".json")
    weights_filedir = os.path.join(root_dir,"models",str(params['data'])+"_model_" + str(params['model_num']) + ".h5")
    model.save_model(json_filedir,weights_filedir)
    
    # Plot information
    roc_filedir = os.path.join(root_dir,'images',params['neg'] + "_" + params['name'] +
                                 "_" + "AUC_" + str(params['model_num']) + "_day" + str(params['day']) +
                                 "_filters" + str(params['filters']) + "_filter_size" +
                                 str(params['filter_size']) +  "_batch" + str(params['batch']) +  "_" +
                                 params['opt'] + "_" + str(params['epoch']) + "epochs" + ".png")    
    PlotStatistics(roc_filedir,tprs, aucs, auprs, mean_fpr)
