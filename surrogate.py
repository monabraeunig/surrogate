## checkpointing with PyTorch
import umbridge
import threading
import torch
import botorch
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from queue import Queue
import os


class Surrogate(umbridge.Model):
    
    ## constructor
    def __init__(self):
        super().__init__("surrogate")
        ## connect to the UM-Bridge model
        self.umbridge_model = umbridge.HTTPModel('http://0.0.0.0:4243', "posterior")
        
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
            self.in_list = torch.empty((0, 2), dtype=torch.double)
            ## Output Data
            self.out_list = torch.empty((0, 1), dtype=torch.double)
            
        ## gathered data for updates
        self.in_queue = Queue()
        self.out_queue = Queue()
        ## Lock
        self.lock = threading.Lock()
        self.pos_lock = threading.Lock()
        ## new data available => gp should be updated
        self.update_data = threading.Event()
        
    def get_input_sizes(self, config):
        return[2]
    
    def get_output_sizes(self, config):
        return[1]

    ## gp gets trained
    def train_gp(self, config):
        with self.lock:
                
            self.in_list = torch.cat([self.in_list] + list(self.in_queue.queue), dim=0)
            self.in_queue = Queue()
            self.out_list = torch.cat([self.out_list]+ list(self.out_queue.queue), dim=0)
            self.out_queue = Queue()

        new_gp = botorch.models.gp_regression.SingleTaskGP(self.in_list, self.out_list, 1e-6*torch.ones_like(self.out_list), outcome_transform=Standardize(self.out_list.shape[1]), input_transform=Normalize(self.in_list.shape[1]))
        
        infoin = torch.vstack([torch.tensor(self.in_list[-1]).flatten()], out=None)
        
        with self.pos_lock:
            with torch.no_grad():
                posterior_new = new_gp.posterior(infoin)
            
        with self.lock:
            self.gp = new_gp
     
    def generate_new_data(self, parameters, config):
        ##model_output
        out = self.umbridge_model(parameters)[0]
        with self.lock:
            self.in_queue.put(torch.tensor(parameters, dtype=torch.double))
            self.out_queue.put(torch.tensor([out], dtype=torch.double))
            self.update_data.set()
        return out
    
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
            if torch.max(pos_variance) > 0.0001:
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
            'output': self.out_list,},
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

umbridge.serve_models([testmodel], 4242; max_workers=1000)
