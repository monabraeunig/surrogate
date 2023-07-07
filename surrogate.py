import umbridge
import time
import os
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import gpytorch
import botorch

from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize

from botorch.fit import fit_gpytorch_mll

from tinyDA.umbridge import UmBridgeModel
from scipy.stats import multivariate_normal
from tinyDA.sampler import*


class Surrogate(umbridge.Model):

    def __init__(self):
        super().__init__("surrogate")
     
        ## connect to the UM-Bridge model
        umbridge_model = umbridge.HTTPModel('http://0.0.0.0:4243', "posterior")
        
        ## wrap the UM-Bridge model in the tinyDA UM-Bridge interface.
        tda = UmBridgeModel(umbridge_model, umbridge_config={'level': 1})
        self.my_model = tda
        
        ## number of times __call__ function was called
        self.init = 0
        ## input
        self.X = 0
        ## output
        self.y = 0
        ## noise
        self.y_var = 0
        ## gaussian process
        self.gp = 0

    def get_input_sizes(self, config):
        return[2]
    
    def get_output_sizes(self, config):
        return[1]
    

    def __call__(self, parameters, config):
          
        number_of_input_vectors = len(self.get_input_sizes(config))
        number_of_output = self.get_output_sizes(config)[0]
        
        ## put parameters in tensor
        get_values = [] 
        get_tensors = [] 
        
        for j in range(number_of_input_vectors):
            get_values.append(torch.tensor(parameters[j]))
            get_tensors.append(get_values[j].flatten())
        
        model_output = np.zeros(number_of_output)
        out = []
        
        ## in case of an output vector with more than one dimension
        var = np.zeros(number_of_output)
        sortvar = np.zeros(number_of_output)
        
        ## contains the parameters to put into gp
        test1 = torch.vstack(get_tensors, out=None)
        
        ## contains the parameters to put into my_model
        ar = np.array(parameters[0])
        
        ## first ever call of this method
        if self.init == 0:
            
            ## model has to calculate
            model_output = self.my_model(ar)
            for p in range(number_of_output):
                out.append(model_output[p])  
            
            ## gp gets trained for the first time
            ## put data into tensor format.
            ## input
            self.X = torch.tensor(parameters, dtype=torch.double)
            ## output
            yy = np.array([out])
            self.y = torch.tensor(yy, dtype=torch.double)
            ## noise
            self.y_var = 1e-16*torch.ones_like(self.y)
        
            ## outputs should be zero mean, unit variance.
            outcome_transform = Standardize(self.y.shape[1])

            ## inputs should be on the unit hypercube.
            input_transform = Normalize(self.X.shape[1])

            ## put it all together in a botorch gp
            self.gp = botorch.models.FixedNoiseGP(self.X, self.y, self.y_var, outcome_transform=outcome_transform, input_transform=input_transform)
            
            self.init = 1
        
        ## gp needs to get 3 sets of input and output data to be able to calculate mean and variance
        elif self.init > 0 and self.init < 3:
            
            model_output = self.my_model(ar)
            for p in range(number_of_output):
                out.append(model_output[p])  
            
            # put the data into tensor format.
            yy = np.array([out])
            new_X = torch.tensor(parameters, dtype=torch.double)
            new_y = torch.tensor(yy, dtype=torch.double)
            
            ## attach old and new data
            self.X = torch.cat([self.X, new_X], dim=0)
            self.y = torch.cat([self.y, new_y], dim=0)
            self.y_var = 1e-16*torch.ones_like(self.y)
                
            outcome_transform = Standardize(self.y.shape[1])
            input_transform = Normalize(self.X.shape[1])

            ## train gp with old and new data
            self.gp = botorch.models.FixedNoiseGP(self.X, self.y, self.y_var, outcome_transform=outcome_transform, input_transform=input_transform)
            
            self.init = self.init + 1
        
        else:
            ## let gp predict the output
            with torch.no_grad():
                posterior2 = self.gp.posterior(test1)
                model_output = posterior2.mean
                var = posterior2.variance
            
                ## find maximum variance 
                for k in range(number_of_output):
                    sortvar[k] = var[0][k].item()
                
                ## if variance is too high model is called
                if np.amax(sortvar) > 0.001 :
                    model_output = self.my_model(ar)
                    for h in range(number_of_output):
                        out.append(model_output[h])       
                
                    ## and data is put in gp 
                    yy = np.array([out])
                    new_X = torch.tensor(parameters, dtype=torch.double)
                    new_y = torch.tensor(yy, dtype=torch.double)
                
                    self.X = torch.cat([self.X, new_X], dim=0)
                    self.y = torch.cat([self.y, new_y], dim=0)
                    self.y_var = 1e-16*torch.ones_like(self.y)
                    
                    outcome_transform = Standardize(self.y.shape[1])
                    input_transform = Normalize(self.X.shape[1])
             
                    self.gp = botorch.models.FixedNoiseGP(self.X, self.y, self.y_var, outcome_transform=outcome_transform, input_transform=input_transform) 
                
                ## if variance is small enough work with prediction og gp
                else :
                    for i in range(number_of_output):
                        out.append(model_output[0][i].item())  
                
        return[out]

    def supports_evaluate(self):
        return True
    
    def gardient(self,out_wrt, in_wrt, parameters, sens, config):
        return [2*sens[0]]
    
    def supports_gardient(self):
        return True
    
    

testmodel = Surrogate()

umbridge.serve_models([testmodel], 4242)
