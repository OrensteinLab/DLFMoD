#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:21:56 2019

@author: yahelsalomon
"""

from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

class Models():
    def __init__(self,input_shape,params):
        self.name = params['name']
        self.input_shape = input_shape
        self.num_classes = params['num_classes']
        self.params = params
        self.model = None  
    def get_model(self):
        return self.model   
    def set_model(self, updated):
        self.model = updated    
    def compile_model(self):
        if self.params['opt']=='adam':
            self.model.compile(loss='binary_crossentropy', optimizer=self.params['opt'], metrics=['accuracy'])
        elif self.params['opt']=='rms':
            self.model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001, decay=1e-4), metrics=['accuracy'])
        self.set_model(self.model)
    def save_model(self,json_filedir,weights_filedir):
        model_json = self.model.to_json()
        with open(json_filedir, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(weights_filedir)
    
class simpleConvModel(Models):
    def __init__(self,input_shape,params):
        super(simpleConvModel,self).__init__(input_shape,params)       
    def create_model(self):
        model = Sequential()
        model.add(Conv1D(self.params['filters'], self.params['filter_size'], strides = 1, activation='relu', input_shape=self.input_shape, use_bias=True))
        model.add(GlobalMaxPooling1D())#model.add(MaxPooling1D(self.params['pool']))#
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        self.set_model(model)
        

         


        
        
        
        
        
        
        