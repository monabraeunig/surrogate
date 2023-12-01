import umbridge
import numpy as np
import threading
import torch
import botorch
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from queue import Queue

import time
from line_profiler import LineProfiler
from viztracer import log_sparse
from viztracer import get_tracer

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
        ## gathered data
        self.in_queue = Queue()
        self.out_queue = Queue()
        ## Lock
        self.lock = threading.Lock()
        self.lock2 = threading.Lock()
        ##new attempt
        ##self.gp wurde geupdatet (posterior noch nicht berechnet)
        self.update_event = threading.Event()
        #neue daten wurden in die in und ou Queue gelegt
        self.update_data = threading.Event()
        self.update_done = threading.Event()
        self.update_done.set()
        self.count = 0
        
    def get_input_sizes(self, config):
        return[2]
    
    def get_output_sizes(self, config):
        return[1]

    ## gp gets trained
    #@profile
    #@log_sparse(stack_depth=1)
    @log_sparse
    def train_gp(self, config):
        with get_tracer().log_event("lock_in_use"):
            with self.lock:
            
                if self.gp is not None:
                    self.in_list = torch.cat([self.in_list] + list(self.in_queue.queue), dim=0)
                    self.in_queue = Queue()
                    self.out_list = torch.cat([self.out_list]+ list(self.out_queue.queue), dim=0)
                    self.out_queue = Queue()

        new_gp = botorch.models.FixedNoiseGP(self.in_list, self.out_list, 1e-6*torch.ones_like(self.out_list), outcome_transform=Standardize(self.out_list.shape[1]), input_transform=Normalize(self.in_list.shape[1]))

        #time.sleep(1)
        infoin = torch.vstack([torch.tensor(self.in_list[-1]).flatten()], out=None)
        with torch.no_grad():
            posterior_ = new_gp.posterior(infoin)
            
        with get_tracer().log_event("lock_in_use"):    
            with self.lock:
                self.gp = new_gp
                self.update_event.set()
        
    #@profile 
    #@log_sparse(stack_depth=1)
    @log_sparse
    def __call__(self, parameters, config):
        if self.gp is None:
            ##model_output
            out = self.umbridge_model(parameters)[0]
            with get_tracer().log_event("lock_in_use"):
                with self.lock:
                    self.in_list = torch.tensor(parameters, dtype=torch.double)
                    self.out_list = torch.tensor([out], dtype=torch.double)
            self.train_gp(config)
            
            return [out]
        
        else:
            ## parameters to put into gp
            infoin = torch.vstack([torch.tensor(param).flatten() for param in parameters], out=None)
            ## let gp predict the output
            ###
            with torch.no_grad():
                time.sleep(0.4)
                posterior_ = self.gp.posterior(infoin)
                pos_variance = posterior_.variance
            ## wir brauchen das nicht mehr glaube ich, außer wir haben dann ein problem mit max variance aber vermutlich ist das dann einfach false
            ## parallel braucht man das eigentlich nicht, sequenziell sollte ich nochmal gucken
            ## ich könnte if torch.max(pos_variance) > 0.01 or torch.any(torch.isnan(pos_variance)): machen ich denke das müsste passen für beide Fälle
            if torch.any(torch.isnan(pos_variance)):
                out = self.umbridge_model(parameters)[0]
                with get_tracer().log_event("lock_in_use"):
                    with self.lock:
                        self.in_queue.put(torch.tensor(parameters, dtype=torch.double))
                        self.out_queue.put(torch.tensor([out], dtype=torch.double))
                        self.update_data.set()
                        
                return [out]
            else:
                    ## if variance is too high model is called
                if torch.max(pos_variance) > 0.01:
                    model_output = self.umbridge_model(parameters)[0]
                    
                    with get_tracer().log_event("lock_in_use"):
                        with self.lock:
                            self.in_queue.put(torch.tensor(parameters, dtype=torch.double))
                            self.out_queue.put(torch.tensor([model_output], dtype=torch.double))
                            self.update_data.set()
                        
                    return [model_output]
                
            pos_mean = posterior_.mean
            print("kein model/train_gp")
            return [pos_mean.flatten().tolist()]
                
    
    def supports_evaluate(self):
        return True
        
    #@profile
    #@log_sparse(stack_depth=2)
    def update_gp_thread(self):
        while True:
            #time.sleep(1)
            self.update_event.wait()
            self.update_event.clear()
            self.update_data.wait()
            self.update_data.clear()
            if not self.in_queue.empty() and not self.out_queue.empty():
                print(self.out_queue.qsize())
                if print(self.out_queue.qsize()) == 1:
                    print(f'inlist: {list(self.out_queue.queue)}')
                self.train_gp(None)   
            #else: 
                #self.update_event.set()
           
                
            
    
        
testmodel = Surrogate()

update_thread = threading.Thread(target=testmodel.update_gp_thread, daemon=True)
update_thread.start()

umbridge.serve_models([testmodel], 4242)
