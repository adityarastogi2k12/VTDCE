#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:00:26 2020

@author: mig-arindam
"""


import numpy as np
import torch

class Config :
    
    def __init__(self):
        

        
        self.gpu       = True
        self.gpuid     = 0
        self.batchsize = 1
        self.optimizer = "Adam"

        
        
        
