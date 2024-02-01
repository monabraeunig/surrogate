import umbridge
import threading
import torch
import gpytorch
import botorch
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from queue import Queue
import os
import json

import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

class Surrogate(umbridge.Model):
    
    ## constructor
    def __init__(self):
        super().__init__("surrogate")
        
        ## load config file
        with open("custom_surrogate.json", "r") as f:
            config = json.load(f)
        
        ## connect to the UM-Bridge model
        self.umbridge_model = umbridge.HTTPModel(config["model_port"], config["model_name"])

        ## get input and output size from umbridge_model
        self.input_size = self.umbridge_model.get_input_sizes(None)[0]
        self.output_size = self.umbridge_model.get_output_sizes(None)[0]
        
        ## strat from checkpoint if existent
        if os.path.exists('checkpoint.pth'):
            ## load checkpoint
            loaded_checkpoint = torch.load('checkpoint.pth')
            ## input data
            self.in_list = loaded_checkpoint['input']
            ## output data
            self.out_list = loaded_checkpoint['output']
            ## initialize Gaussian Process
            self.gp = botorch.models.gp_regression.SingleTaskGP(self.in_list, self.out_list, 1e-6*torch.ones_like(self.out_list), outcome_transform=Standardize(self.out_list.shape[1]), input_transform=Normalize(self.in_list.shape[1]))
            self.gp.load_state_dict(loaded_checkpoint['model_state_dict'])
        else:
            self.checkpoint_file = 'checkpoint.pth'
            ## Gaussian Process
            self.gp = None
            ## Input Data
            self.in_list = torch.empty((0, self.input_size), dtype=torch.double)
            ## Output Data
            self.out_list = torch.empty((0, self.output_size), dtype=torch.double)

        ## set Hyperparameters
        # set lengthscale
        self.custom_lengthscale = torch.tensor(config["lengthscale"], dtype=torch.float64)
        # set output scale
        self.custom_outputscale = torch.tensor(config["outputscale"], dtype=torch.float64)
        # set mean
        self.custom_mean = torch.tensor(config["mean"], dtype=torch.float64)

        ## set threshold
        self.threshold = config["threshold"]

        ## gathered training data for
        self.in_queue = Queue()
        self.out_queue = Queue()
        ## Lock
        self.lock = threading.Lock()
        self.pos_lock = threading.Lock()
        ## new data available => gp should be updated
        self.update_data = threading.Event()

        ## plot variance
        self.plot_enabled = config["plot"]
        if self.plot_enabled:
            ## varianz plotten config file könnte man einstellen und hier dann mit if
            self.lower_bound = [-6.0, -6.0]
            self.upper_bound = [6.0, 6.0]
            ## overall number of plots
            self.init = 0
            self.frames = []
        
    def get_input_sizes(self, config):
        return[self.input_size]
        
    def get_output_sizes(self, config):
        return[self.output_size]
    
    ## gp gets trained
    def train_gp(self, config):
        
        with self.lock:
            if self.plot_enabled:
                points = [tensor.squeeze().tolist() for tensor in self.in_queue.queue]
            self.in_list = torch.cat([self.in_list] + list(self.in_queue.queue), dim=0)
            self.in_queue = Queue()
            self.out_list = torch.cat([self.out_list]+ list(self.out_queue.queue), dim=0)
            self.out_queue = Queue()
        
        new_gp = botorch.models.gp_regression.SingleTaskGP(self.in_list, self.out_list, 1e-6*torch.ones_like(self.out_list), outcome_transform=Standardize(self.out_list.shape[1]), input_transform=Normalize(self.in_list.shape[1]))

        new_gp.covar_module.base_kernel.raw_lengthscale.data = self.custom_lengthscale
        new_gp.covar_module.raw_outputscale.data = self.custom_outputscale
        new_gp.mean_module.raw_constant.data = self.custom_mean
        
        infoin = torch.vstack([torch.tensor(self.in_list[-1]).flatten()], out=None)
        
        with self.pos_lock:
            with torch.no_grad():
                posterior_new = new_gp.posterior(infoin)
    
        self.gp = new_gp
        self.gp.eval()

        #time.sleep(1)
        if self.plot_enabled:
            self.init = self.init + 1
            self.heatmap(config, points)
     
    def generate_new_data(self, parameters, config):
        ## calculate output of umbridge_model
        out = self.umbridge_model(parameters)[0]
        with self.lock:
            self.in_queue.put(torch.tensor(parameters, dtype=torch.double))
            self.out_queue.put(torch.tensor([out], dtype=torch.double))
            self.update_data.set()
        return out

    def heatmap(self, config, points):
        ## von -6 bis 6 100 Datenpunkte
        a = np.linspace(self.lower_bound[0], self.upper_bound[0], 100)
        b = np.linspace(self.lower_bound[1], self.upper_bound[1], 100)
        ## generate meshgrid
        A, B = np.meshgrid(a, b) ## das hier sind zwei meshgrids... warum?
        grid_points = np.column_stack((A.ravel(), B.ravel()))
        ## put meshrid into tensor
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32)

        ## let gp calculate variance for entire meshgrid
        with self.pos_lock:
            with torch.no_grad():
                predictions = self.gp.posterior(grid_points_tensor)
                mean = predictions.mean 
                variance = predictions.variance
        
        # Reshape mean and variance to match the grid shape
        mean = mean.reshape(A.shape) ## vermutlich unnötig
        variance = variance.reshape(B.shape)
        
        ## plot
        plt.figure()
        # ajust colormap
        plt.contourf(A, B, variance, levels=20, cmap='viridis')

        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]

        # Plot the data points with red 'x' markers
        plt.scatter(x_values, y_values, c='red', marker='x')
        
        plt.colorbar()
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('GP Variance Visualization')
        plt.savefig(f'gpvar_{self.init}.png')
        self.frames.append(f'gpvar_{self.init}.png')
    
    def __call__(self, parameters, config):
        
        if self.gp is None:
            out = self.generate_new_data(parameters, config)
            return [out]
        
        else:
            ## parameters to put into gp
            infoin = torch.vstack([torch.tensor(param).flatten() for param in parameters], out=None)
            ## let gp predict the output
            with self.pos_lock:
                with torch.no_grad():
                    posterior_ = self.gp.posterior(infoin)
            with torch.no_grad():
                pos_variance = posterior_.variance
                
            ## if variance is too high model is called
            if torch.max(pos_variance) > self.threshold:
            #if torch.max(pos_variance) > 0.0001:    
                model_output = self.generate_new_data(parameters, config)
                return[model_output]
                        
            pos_mean = posterior_.mean
            return [pos_mean.flatten().tolist()]
                
    
    def supports_evaluate(self):
        return True

    def save_checkpoint (self):
        torch.save({
            'model_state_dict': self.gp.state_dict(),
            'input': self.in_list,
            'output': self.out_list},
                   'checkpoint.pth')
            
    def update_gp_thread(self):
        while True:
            self.update_data.wait()
            self.update_data.clear()
            if self.gp is None:
                if not self.out_queue.qsize() < 3:
                    self.train_gp(None)
                    self.save_checkpoint()
            else:
                if not self.in_queue.empty() and not self.out_queue.empty():
                    self.train_gp(None)
                    self.save_checkpoint()
           
testmodel = Surrogate()

update_thread = threading.Thread(target=testmodel.update_gp_thread, daemon=True)
update_thread.start()

umbridge.serve_models([testmodel], 4242)
