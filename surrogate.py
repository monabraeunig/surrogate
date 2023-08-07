import umbridge
import numpy as np
import threading
import torch
import botorch
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize

class Surrogate(umbridge.Model):
    
    ## constructor
    def __init__(self):
        super().__init__("surrogate")
        ## connect to the UM-Bridge model
        self.umbridge_model = umbridge.HTTPModel('http://0.0.0.0:4243', "posterior")
        ## gaussian process
        self.gp = None
        ## Input data
        self.in_list = None
        ## Lock
        self.lock = threading.Lock()

    def get_input_sizes(self, config):
        return[2]
    
    def get_output_sizes(self, config):
        return[1]

    ## gp gets trained
    def train_gp(self, config, inpt, output):
        if self.in_list == None :
            self.in_list = torch.tensor(inpt, dtype=torch.double)
            self.out_list = torch.tensor(output, dtype=torch.double)
        elif len(self.in_list) == 1 :
            self.in_list = torch.cat([self.in_list, inpt], dim=0)
            self.out_list = torch.cat([self.out_list, output], dim=0)    
        else :    
            self.in_list = torch.cat([self.in_list, inpt], dim=0)
            self.out_list = torch.cat([self.out_list, output], dim=0)
            
            noise = 1e-16*torch.ones_like(self.out_list)
        
            outcome_transform = Standardize(self.out_list.shape[1])
            input_transform = Normalize(self.in_list.shape[1])
        
            self.gp = botorch.models.FixedNoiseGP(self.in_list, self.out_list, noise, outcome_transform=outcome_transform, input_transform=input_transform)
        
    def __call__(self, parameters, config):
        ## gp needs to get 3 sets of input and output data to be able to calculate mean and variance
        if self.gp == None:
            model_output = self.umbridge_model(parameters)[0]
            out = model_output
            
            self.lock.acquire()
            self.train_gp(config, torch.tensor(parameters, dtype=torch.double), torch.tensor(np.array([model_output]), dtype=torch.double))
            self.lock.release()
        
        else:
            ## parameters to put into gp
            input_vectors = [torch.tensor(param).flatten() for param in parameters]
            infoin = torch.vstack(input_vectors, out=None)
            ## let gp predict the output
            with torch.no_grad():
                posterior_ = self.gp.posterior(infoin)
                pos_mean = posterior_.mean
                pos_variance = posterior_.variance
            
                ## find maximum variance 
                number_of_output = self.get_output_sizes(config)[0]
                sortvar = np.zeros(number_of_output)
                for k in range(number_of_output):
                    sortvar[k] = pos_varinace[0][k].item()
                
                ## if variance is too high model is called
                if np.amax(sortvar) > 0.0001 :
                    model_output = self.umbridge_model(parameters)[0]
                    
                    self.lock.acquire()
                    self.train_gp(config, torch.tensor(parameters, dtype=torch.double), torch.tensor(np.array([model_output]), dtype=torch.double))
                    self.lock.release()

                    ## gp predicts again
                    with torch.no_grad():
                        posterior_ = self.gp.posterior(infoin)
                        pos_mean = posterior_.mean
                
                out = []
                for i in range(number_of_output):
                    out.append(pos_mean[0][i].item())  
                
        return[out]
    
    def supports_evaluate(self):
        return True

testmodel = Surrogate()

umbridge.serve_models([testmodel], 4242)
