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

        ## True => no fitting / False => fitting
        self.custom_hyperparameters = config["custom_hyperparameters"]
        
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
            
            if self.custom_hyperparameters:
                ##load hyperparameters from config file
                self.custom_lengthscale = torch.tensor(config["lengthscale"], dtype=torch.float64)
                self.custom_outputscale = torch.tensor(config["outputscale"], dtype=torch.float64)
                self.custom_mean = torch.tensor(config["mean"], dtype=torch.float64)
            else:
                ## load Hyperparameters from checkpoint
                self.custom_lengthscale = loaded_checkpoint['lengthscale']
                self.custom_outputscale = loaded_checkpoint['outputscale']
                self.custom_mean = loaded_checkpoint['mean']
                ## load next required fit from checkpoint
                self.next_fit = loaded_checkpoint['next_fit']
                ## number of total fits
                self.its = round(self.next_fit ** (1/3))
                
            ## set Hyperparameters for gp
            self.gp.covar_module.base_kernel.raw_lengthscale.data = self.custom_lengthscale
            self.gp.covar_module.raw_outputscale.data = self.custom_outputscale
            self.gp.mean_module.raw_constant.data = self.custom_mean 

            ## number of saved observations (for checkpointing)
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
            ## set hyperparameters if existent
            if self.custom_hyperparameters:
                self.custom_lengthscale = torch.tensor(config["lengthscale"], dtype=torch.float64)
                self.custom_outputscale = torch.tensor(config["outputscale"], dtype=torch.float64)
                self.custom_mean = torch.tensor(config["mean"], dtype=torch.float64)
            ## if no hyperparameters are available set up for fitting    
            else:
                ## next required fit 
                self.next_fit = 1
                ## number of total fits 
                self.its = 0
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

        ## to stop fitting after model call ratio is below 1/100
        ## number of surrogate calls
        self.count_scalls = 1
        ## number of model calls
        self.count_mcalls = 1
        ## lock for the both to avoid race condition
        self.count_lock = threading.Lock()

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

        ## do this if fitting of the hyperparameters is required
        if not self.custom_hyperparameters and self.count_mcalls/self.count_scalls >= 0.01 and (self.gp is None or self.next_fit <= len(self.out_list)):
            ## reset surrogate and model call counters
            with self.count_lock:
                self.count_scalls = 1
                self.count_mcalls = 1
            self.its = self.its + 1
            ## in case that more observations are available than needed for the next step, skip to next required step
            while self.its ** 3 <= len(self.out_list):
                self.its += 1   
            ## next fitting happens whenever a total of self.next_fit observations are available
            self.next_fit = self.its**3
            ## perform fitting
            with self.pos_lock:
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(new_gp.likelihood, new_gp)
                fit_gpytorch_mll(mll)
            ## put hyperparameter value into class instances for future training without a fitting
            self.custom_lengthscale = new_gp.covar_module.base_kernel.raw_lengthscale.data
            self.custom_outputscale = new_gp.covar_module.raw_outputscale.data
            self.custom_mean = new_gp.mean_module.raw_constant.data
        ## do this when no fitting is required
        else:
            ## assign current hyperparameter values to the new gp
            new_gp.covar_module.base_kernel.raw_lengthscale.data = self.custom_lengthscale
            new_gp.covar_module.raw_outputscale.data = self.custom_outputscale
            new_gp.mean_module.raw_constant.data = self.custom_mean
            
        ## make first (expensive) posterior calculation
        infoin = torch.vstack([self.in_list[-1].flatten()], out=None)
        with self.pos_lock:
            with torch.no_grad():
                posterior_new = new_gp.posterior(infoin)
                
        ## hand over new posterior to gp we are working with
        self.gp = new_gp
        ## put gp into evaluation mode
        self.gp.eval()

        if self.plot_enabled:
            self.init = self.init + 1
            self.heatmap(config, points)
        

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
                
            ## check if uncertainty is to high 
            if torch.max(pos_variance) > self.threshold:
                ## if uncertainty is to high, let UM-Bridge calculate the output and return it
                with self.count_lock:
                    self.count_scalls += 1
                    self.count_mcalls += 1
                model_output = self.generate_new_data(parameters, config)
                return model_output

            ## if uncertainty is low enough don't call UM-Bridge model
            with self.count_lock:
                self.count_scalls += 1 
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
        ## if hyperparameters are specified in the configuration file, they can be accessed from there 
        if self.custom_hyperparameters:
            torch.save({
                'input': self.in_list,
                'output': self.out_list,
                'total_time': self.total_time},
                       'checkpoint.pth')
        ## if hyperparameters are calculated during fitting they need to be saved as well as the observations
        else:
            torch.save({
                'input': self.in_list,
                'output': self.out_list,
                'lengthscale': self.custom_lengthscale,
                'outputscale': self.custom_outputscale,
                'mean': self.custom_mean,
                'next_fit': self.next_fit,
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
