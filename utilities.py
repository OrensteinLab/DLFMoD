# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 12:13:30 2020

@author: yahelsalomon
"""
import sys
import numpy as np
from sklearn.model_selection import KFold
from scipy import interp
import matplotlib.pyplot as plt

from sklearn.metrics import auc
import statistics

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score

def CrossValidationTraining(root_dir,x_data,y_data,params):
    '''
    Perform 10-fold cross-validation on the data.
    
    Args:
        root_dir : Path to the main directory.
        x_data   : One-hot encoded sequences dataset.
        y_data   : Sequences lables.
        params   : parameters information.
        
    Returns:
         model      : Model object.
         tprs       : An array of TPR scores. 
         aucs       : An array of AUC scores. 
         auprs      : An array of AUPR scores. 
         mean_fpr   : Mean of FPR. 
    '''
    n_samples, nx, ny = x_data.shape
    model = None
    train_dict = {'nx':nx,'ny':ny,'verb':2}
    cv1 = KFold(n_splits=10,shuffle=True)
    
    mean_fpr = np.linspace(0, 1, x_data.shape[0])
    tprs = []; aucs = []; auprs = []
    
    plt.figure(figsize = (10,5))
    
    for iter_num,(train, test) in enumerate(cv1.split(x_data, y_data)):
        model = InitializeModel(params['num_classes'],train_dict['nx'],train_dict['ny'],params)
        keras_model = model.get_model()
        _ = keras_model.fit(x_data[train], y_data[train],
                                  batch_size=params['batch'],
                                  epochs=params['epoch'],
                                  verbose=train_dict['verb'],
                                  shuffle=True)
        tprs,aucs,auprs = RocCurve(keras_model,x_data[test],y_data[test],mean_fpr,tprs,aucs,auprs,iter_num)         
    model.set_model(keras_model)    
    return model, tprs, aucs, auprs, mean_fpr 

def StraightforwardTraining(root_dir,x_train,y_train,x_test,y_test,params):
    '''
    Perform strightforward training using train set and returns the parameters needed for
    statistics calculations.
    
    Args:
        root_dir    : Path to the main directory.
        x_train     : One-hot encoded sequences training set.
        y_train     : Sequences lables of the training set.
        x_test      : One-hot encoded sequences test set.
        y_test      : Sequences lables of the test set.
        params      : parameters information.
        
    Returns:
         model      : Model object.
         tprs       : An array of TPR scores. 
         aucs       : An array of AUC scores. 
         auprs      : An array of AUPR scores. 
         mean_fpr   : Mean of FPR. 
    '''
    n_samples, nx, ny = x_train.shape
    model = None
    train_dict = {'nx':nx,'ny':ny,'verb':2}
    
    mean_fpr = np.linspace(0, 1, x_train.shape[0])
    tprs = []; aucs = []; auprs = []
    
    plt.figure(figsize = (10,5))
    
    model = InitializeModel(params['num_classes'],train_dict['nx'],train_dict['ny'],params)
    keras_model = model.get_model()
    _ = keras_model.fit(x_train,y_train,
                              batch_size=params['batch'],
                              epochs=params['epoch'],
                              verbose=train_dict['verb'],
                              shuffle=True)
    tprs,aucs,auprs = RocCurve(keras_model,x_test,y_test,mean_fpr,tprs,aucs,auprs,0) 
    model.set_model(keras_model)
    return model, tprs, aucs, auprs, mean_fpr 

# String to class converter
def Str2Class(str,args):
    return getattr(sys.modules[__name__], str)(*args)
    
def InitializeModel(nx,ny,params):
    '''
    Initializing Model object.
    
    Args:
        nx      : Sequence length.
        ny      : 4.
        params  : parameters information.
        
    Returns:
         model  : initial Model object.
    '''
    input_shape = (nx,ny)
    args = [params['name'],input_shape, params['num_classes'],params]
    model = Str2Class(params['name'],args)
    model.create_model()
    model.compile_model()
    return model

def RocCurve(model,x_data,y_data,mean_fpr,tprs,aucs,auprs,iter_num):
    '''
    Plot the Roc Curve.
    
    Args:
        model       : Trained keras_model
        x_data      : One-hot encoded sequences dataset.
        y_data      : Sequences lables.
        mean_fpr    : Mean of FPR
        tprs        : An array of TPR scores. 
        aucs        : An array of AUC scores. 
        auprs       : An array of AUPR scores.
        iter_num    : Fold counter.
        
    Returns:
        tprs        : An array of TPR scores. 
        aucs        : An array of AUC scores. 
        auprs       : An array of AUPR scores.
    '''
    predict = model.predict(x_data)
    fpr, tpr, thresholds = roc_curve(y_data, predict, pos_label=1)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (iter_num, roc_auc))    
    aupr = average_precision_score(y_data, predict)
    print("aupr = ", aupr)
    auprs.append(aupr)
    return tprs, aucs, auprs

def PlotStatistics(roc_filedir,tprs, aucs, auprs, mean_fpr): 
    '''
    Plot Roc Curves of 10-fold cross-validation and save graph.
    
    Args:
        roc_filedir     : Path to the roc curves graph directory. 
        tprs            : An array of TPR scores. 
        aucs            : An array of AUC scores. 
        auprs           : An array of AUPR scores.
        mean_fpr        : Mean of FPR.
        
    Returns:
        tprs        : An array of TPR scores. 
        aucs        : An array of AUC scores. 
        auprs       : An array of AUPR scores.
    '''
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Random', alpha=.8)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    plt.savefig(roc_filedir)
    
    mean_aupr = np.mean(auprs)
    mean_auc = np.mean(aucs) 
    print("mean_aupr = ",mean_aupr)
    print("std of aupr is % s " % (statistics.stdev(auprs)))
    print("mean_auc = ",mean_auc)
    print("std of auc is % s " % (statistics.stdev(aucs)))
    