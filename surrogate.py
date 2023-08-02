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
        
        ## number of times __call__ function was called
        self.init = 0
        ## gaussian process
        self.gp = 0

        self.lock = threading.Lock()

    def get_input_sizes(self, config):
        return[2]
    
    def get_output_sizes(self, config):
        return[1]
    

    ## model calculates the output 
    def model_calc(self, config, values):
        number_of_output = self.get_output_sizes(config)[0]
        model_output = np.zeros(number_of_output)
        out = []
        
        model_output = self.umbridge_model(values)
        for i in range(number_of_output):
            out.append(model_output[i])
            
        return out[0]

    ## gp calculates the output
    def train_gp(self, config, inpt, output, boole):
        if boole == 0 :
            self.X = torch.tensor(inpt, dtype=torch.double)
            self.y = torch.tensor(output, dtype=torch.double)
            
        else :    
            self.X = torch.cat([self.X, inpt], dim=0)
            self.y = torch.cat([self.y, output], dim=0)
            
        self.y_var = 1e-16*torch.ones_like(self.y)
        
        self.outcome_transform = Standardize(self.y.shape[1])
        self.input_transform = Normalize(self.X.shape[1])
        
        self.gp = botorch.models.FixedNoiseGP(self.X, self.y, self.y_var, outcome_transform=self.outcome_transform, input_transform=self.input_transform)
        
    
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
        
        ## in case of an output vector with more than one dimension
        var = np.zeros(number_of_output)
        sortvar = np.zeros(number_of_output)
        
        ## contains the parameters to put into gp
        infoin = torch.vstack(get_tensors, out=None)
        
        ## first ever call of this method
        if self.init == 0:
            ## model has to calculate
            out = self.model_calc(config, parameters)

            ## gp is trained for the first time
            self.lock.acquire()
            self.train_gp(config, torch.tensor(parameters, dtype=torch.double), torch.tensor(np.array([out]), dtype=torch.double), 0)
            self.lock.release()
            
            self.init = 1
        
        ## gp needs to get 3 sets of input and output data to be able to calculate mean and variance
        elif self.init > 0 and self.init < 3:
            out = self.model_calc(config, parameters)
            
            self.lock.acquire()
            self.train_gp(config, torch.tensor(parameters, dtype=torch.double), torch.tensor(np.array([out]), dtype=torch.double), 1)
            self.lock.release()
            
            self.init = self.init + 1
        
        else:
            ## let gp predict the output
            with torch.no_grad():
                posterior_ = self.gp.posterior(infoin)
                model_output = posterior_.mean
                var = posterior_.variance
            
                ## find maximum variance 
                for k in range(number_of_output):
                    sortvar[k] = var[0][k].item()
                
                ## if variance is too high model is called
                if np.amax(sortvar) > 0.0001 :
                    outp = self.model_calc(config, parameters)
                    
                    self.lock.acquire()
                    self.train_gp(config, torch.tensor(parameters, dtype=torch.double), torch.tensor(np.array([outp]), dtype=torch.double), 1)
                    self.lock.release()

                    ## given output still comes from gp
                    with torch.no_grad():
                        posterior_ = self.gp.posterior(infoin)
                        model_output = posterior_.mean
                
                out = []
                for i in range(number_of_output):
                    out.append(model_output[0][i].item())  
                
        return[out]
    
    def supports_evaluate(self):
        return True

testmodel = Surrogate()

umbridge.serve_models([testmodel], 4242)
