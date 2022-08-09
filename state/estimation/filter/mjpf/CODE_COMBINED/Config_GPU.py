# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 09:45:35 2021

@author: asus
"""

import torch

# Function to retrieve if GPU or CPU to use
def DefineDeviceVariable(configGPU):
    
    # GPU or CPU?
    if configGPU['force_cpu_use'] == True:
        # This is to force CPU or GPU usage
        device = torch.device("cpu")
    else:
        # This automatically chooses the device that is available:
        # - if there is a GPU, it considers it
        # - otherwise, it takes the CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    return device

# Define if you want to force cpu usage
def ConfigureGPUSettings():
    
    configGPU = {           
            # Do you want to force the use of the cpu?
            "force_cpu_use"            : False}
            
    return configGPU