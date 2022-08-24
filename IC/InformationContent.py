# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 10:02:11 2020

@author: yahelsalomon
"""

import numpy as np
import pandas as pd
import os
from scipy.stats import entropy

def savePattern(patten,filename,LEN=70):
    raw_data = {'Pos':np.arange(LEN)+1,'A': patten[:,0],'C': patten[:,1],'G': patten[:,2],'T': patten[:,3]}
    df = pd.DataFrame(raw_data, columns = ['Pos','A','C','G','T'])
    np.savetxt(filename, df.values, fmt='%i\t%0.6f\t%0.6f\t%0.6f\t%0.6f', delimiter="\t", header="Pos\tA\tC\tG\tT",comments='')

def refineMotif(seq):
    IC = np.array([2-entropy(seq[i],base=2) for i in range(len(seq))]) #np.sum(seq[i]*np.log2(seq[i]))
    IC_refined = np.ones(len(seq)).astype(int)
    for i in range(len(seq)):
        if IC[i]>=0.3: #0.35
            IC_refined[i]=0
    i=0;j=len(IC_refined)-1 
    start=-1; end=-1
    if np.unique(IC_refined).all():
        return start,end
    IC_refined ^= 1
    while(i<j):
        if IC_refined[i]==1 and IC_refined[j]==0:
            start=i;j-=1
        elif IC_refined[i]==0 and IC_refined[j]==1:
            i+=1;end=j
        elif IC_refined[i]==1 and IC_refined[j]==1:
            start=i;end=j
            break
        else:
            i+=1;j-=1
    return start,end+1

rootdir = os.getcwd()

filedir=os.path.join(rootdir) 
outdir=os.path.join(rootdir)
for file in os.listdir(filedir):
    if file.startswith('onehot') and file.endswith('.txt'):
        pattern=np.loadtxt(os.path.join(filedir,file),dtype=np.float,skiprows=1)
        i,j=refineMotif(pattern[:,1:])
        savePattern(pattern[i:j,1:],outdir+"\\refined_"+file,len(pattern[i:j,1:]))
        
