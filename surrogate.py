import umbridge
import threading
import torch
import gpytorch
import botorch
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from botorch.fit import fit_gpytorch_mll
from queue import Queue
import os
import json
import time

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

        ## load configuration file
        with open("custom_surrogate.json", "r") as f:
            config = json.load(f)
        
        ## connect to the UM-Bridge model
        self.umbridge_model = umbridge.HTTPModel(config["model_port"], config["model_name"])

        ## get input and output size from UM-Bridge Model
        self.input_size = self.umbridge_model.get_input_sizes(None)
        self.output_size = self.umbridge_model.get_output_sizes(None)

         ##load hyperparameters from config file
        self.custom_lengthscale = config["lengthscale"]
        self.custom_outputscale = config["outputscale"]
        self.custom_mean = config["mean"]
        
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
            
            ## set Hyperparameters for gp
            self.gp.covar_module.base_kernel.lengthscale = self.custom_lengthscale
            self.gp.covar_module.outputscale = self.custom_outputscale
            self.gp.mean_module.constant = self.custom_mean 

            ## checkpointing
            ## number of saved observations
            self.old_save_size = len(self.out_list)
            ## total time spend with model calculations so far
            self.total_time = loaded_checkpoint['total_time']
            
        ## start from scratch if checkpoint is not existent
        else:
            ## Gaussian Process
            self.gp = None
            ## empty list for input training data
            self.in_list = torch.empty((0, sum(self.input_size)), dtype=torch.double)
            ## empty list for output training data
            self.out_list = torch.empty((0, sum(self.output_size)), dtype=torch.double)
                
            ## checkpointing
            ## number of saved observations (for checkpointing)
            self.old_save_size = 1
            ## total time spend with model calculations so far
            self.total_time = 0
        
        ## new observations
        ## gathered new observation data
        self.in_queue = Queue()
        self.out_queue = Queue()
        ## Lock for observation lists and queues
        self.lock = threading.Lock()
        ## new data available => gp should be updated
        self.update_data = threading.Event()

        ## Lock for posterior calculation
        self.pos_lock = threading.Lock()
        ## threshold for the variance
        self.threshold = config['threshold']

        ## checkpointing dependent on model calculation time and checkpoint saving time
        ## time to save one observation
        self.single_check = (1e-5/2) * (self.in_list.size()[1] + self.out_list.size()[1])
        ## number of unsaved observation data (to know how long calculating these new observations took)
        self.not_saved_data = 0
        ## lock for model time tracking
        self.model_time_lock = threading.Lock()


        ## plot variance
        self.plot_enabled = config["plot"]
        if self.plot_enabled:
            ## varianz plotten config file könnte man einstellen und hier dann mit if
            self.lower_bound = [config['lower_bound_x'], config['lower_bound_y'] ]
            self.upper_bound = [config['upper_bound_x'], config['upper_bound_y']]
            ## overall number of plots
            self.init = 0
            self.frames = []

        
    def get_input_sizes(self, config):
        return self.input_size
        
    def get_output_sizes(self, config):
        return self.output_size

    ## plot variance
    def heatmap(self, config, points):
        a = np.linspace(self.lower_bound[0], self.upper_bound[0], 100)
        b = np.linspace(self.lower_bound[1], self.upper_bound[1], 100)
        ## generate meshgrid
        A, B = np.meshgrid(a, b) 
        grid_points = np.column_stack((A.ravel(), B.ravel()))
        ## put meshrid into tensor
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32)

        ## let gp calculate variance for entire meshgrid
        with self.pos_lock:
            with torch.no_grad():
                predictions = self.gp.posterior(grid_points_tensor)
                variance = predictions.variance
        
        # Reshape variance to match the grid shape
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
    
    ## reinitialize gp and fit the hyperparameters
    def train_gp(self, config):
        ## append new observation data to old trainig data
        with self.lock:
            if self.plot_enabled:
                points = [tensor.squeeze().tolist() for tensor in self.in_queue.queue]
            self.in_list = torch.cat([self.in_list] + list(self.in_queue.queue), dim=0)
            self.in_queue = Queue()
            self.out_list = torch.cat([self.out_list]+ list(self.out_queue.queue), dim=0)
            self.out_queue = Queue()

        ## set up a new gp with all observations 
        new_gp = botorch.models.gp_regression.SingleTaskGP(self.in_list, self.out_list, 1e-6*torch.ones_like(self.out_list), outcome_transform=Standardize(self.out_list.shape[1]), input_transform=Normalize(self.in_list.shape[1]))

        ## set hyperparameters
        new_gp.covar_module.base_kernel.lengthscale = self.custom_lengthscale
        new_gp.covar_module.outputscale = self.custom_outputscale
        new_gp.mean_module.constant = self.custom_mean
            
        ## make first (expensive) posterior calculation
        infoin = torch.vstack([self.in_list[-1].flatten()], out=None)
        with self.pos_lock:
            with torch.no_grad():
                posterior_new = new_gp.posterior(infoin)
                
        ## hand over new posterior to gp we are working with
        self.gp = new_gp
        ## put gp into evaluation mode
        self.gp.eval()

        ## plot variance
        if self.plot_enabled:
            self.init = self.init + 1
            self.heatmap(config, points)

        ## nur im zu prüfen ob auch alles richtig läuft muss aber später raus
        print(self.gp.mean_module.constant)
        print(self.gp.covar_module.outputscale)
        print(self.gp.covar_module.base_kernel.lengthscale)
        

    ## generate new observations
    def generate_new_data(self, parameters, config):
        ## track model calcumation time
        start_time = time.time()
        ## let UM-Bridge model calculate the output
        model_output = self.umbridge_model(parameters)
        ## calculate time model calculation took
        elapsed_time = time.time() - start_time
        with self.lock:
            ## put observation into respective queue
            self.in_queue.put(torch.tensor([[item for sublist in parameters for item in sublist]], dtype=torch.double))
            self.out_queue.put(torch.tensor([[item for sublist in model_output for item in sublist]], dtype=torch.double))
            with self.model_time_lock:
                self.total_time += elapsed_time
                self.average_time = self.total_time/(len(self.out_list) + self.out_queue.qsize())
        ## signal that new observation data is available
        self.update_data.set()
        return model_output

    
    ## process input requests
    def __call__(self, parameters, config):
        ## if gp is not initialized yet (due to a lack of observation data)
        if self.gp is None:
            ## let UM-Bridge calculate the output
            model_output = self.generate_new_data(parameters, config)
            return model_output
        else:
            ## prepare parameters to put into gp
            infoin = torch.tensor([[item for sublist in parameters for item in sublist]])
            ## let gp calculate the posterior for the given parameters
            with self.pos_lock:
                with torch.no_grad():
                    posterior_ = self.gp.posterior(infoin)
            ## get the gps uncertainty about the predicting
            with torch.no_grad():
                pos_variance = posterior_.variance

            #print(torch.max(pos_variance))
            ## check if uncertainty is to high 
            if torch.max(pos_variance) > self.threshold:
                ## if uncertainty is to high, let UM-Bridge calculate the output and return it
                model_output = self.generate_new_data(parameters, config)
                return model_output

            ## if uncertainty is low enough don't call UM-Bridge model
            ## let gp calculate predictive mean
            with torch.no_grad():
                pos_mean = posterior_.mean
            mean = pos_mean.flatten().tolist()
            ## create list of correct size to put output values in to
            gp_output = [[0] * size for size in self.output_size]
            count = 0
            for i in range(len(self.output_size)):
                for j in range(self.output_size[i]):
                    gp_output[i][j] = mean[count]
                    count += 1
            ## return the predictive mean
            return gp_output
                
    
    def supports_evaluate(self):
        return True

    ## save observation data and hyperparameters to a checkpoint file
    def save_checkpoint (self):
        ## number of observatins being saved right now
        self.old_save_size = len(self.out_list)
        torch.save({
            'input': self.in_list,
            'output': self.out_list,
            'total_time': self.total_time},
                    'checkpoint.pth')
        
            
    ## thread responsible to update the gp whenever new observation data is available       
    def update_gp_thread(self):
        while True:
            ## wait until new observation data is available
            self.update_data.wait()
            self.update_data.clear()
            ## if gp has not been initialized yet at least 3 observations need to be available 
            if self.gp is None:
                if not self.out_queue.qsize() < 3:
                    self.train_gp(None)
                    self.save_checkpoint()
            else:
                if not self.in_queue.empty() and not self.out_queue.empty():
                    self.train_gp(None)
                    if self.average_time * (len(self.out_list)-self.old_save_size) > self.single_check * len(self.out_list):
                        self.save_checkpoint()
                        
                       

           
testmodel = Surrogate()

update_thread = threading.Thread(target=testmodel.update_gp_thread, daemon=True)
update_thread.start()


umbridge.serve_models([testmodel], 4242, max_workers=1000)
