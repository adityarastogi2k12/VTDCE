#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:07:48 2020

@author: mig-arindam
"""
import os
abspath = os.path.abspath(ENTER_YOUR_DIRECTORY_PATH) ## String which contains absolute path to the script file
os.chdir(abspath) ## Setting up working directory
#%%
#os,time
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
from datetime import datetime
from config_test import Config
from mydataset_test import myDataset
import h5py as h5
import scipy.io as sio

from model_vtnet_2 import vt_net
import torch.optim as optim


#%%
print ('*************************************************')


config  = Config()
# config.gpu = False

# load the testing data
def getTestingData():
    #num: set this value between 0 to 163. There are total testing 164 slices in testing data
    print('Reading the data. Please wait...')
    filename ='Dataset/R20TestdatasetVTDCE_PATB.hdf5' #set the correct path here
    
    with h5.File(filename,'r') as f:
    	tstData,tstLabel,MapsUS=f['tstData'][:],f['tstLabel'][:],f['MapUS'][:]
#    tstData = tstData[:,0:30,:,:]
    print(tstData.shape)



    return tstData,tstLabel,MapsUS


tstData,tstLabel,MapsUS = getTestingData()





# make the data iterator for testing data
test_data = myDataset(tstData)
testloader  = torch.utils.data.DataLoader(test_data, config.batchsize, shuffle=False, num_workers=2)

print('###########################')
#%%
#Create the object for the network
if config.gpu == True:
    #net = ldct_32d_net()
    net = vt_net()
    #net = NestedUNet()
    net.cuda(config.gpuid)
    par = torch.nn.DataParallel(net, device_ids = [0])
else:
      net = vt_net()
      print('*')



        
    
#%%
net.load_state_dict(torch.load('/checkpoint.pth')) #location of epoch


net = net.eval()
#
Loss_t = []
Kt = []
Vp = []
for i,data in enumerate(testloader): 
    # start iterations
    images = data[0]

    images = images.type(torch.FloatTensor)
  
        
    images = images.unsqueeze(0)
    images = images.unsqueeze(0)
   
    if config.gpu == True:
          images = images.cuda(config.gpuid)
          #labels = labels.cuda(config.gpuid)
    # make forward pass      
    output_test1,output_test2 = net(images)

    
    test_residue1 = output_test1.cpu()
    test_residue1 = test_residue1.detach().numpy()
    Kt.append(test_residue1)
    
    test_residue2 = output_test2.cpu()
    test_residue2 = test_residue2.detach().numpy()
    Vp.append(test_residue2)



plot1= lambda x: plt.imshow(x,cmap=plt.cm.hot, vmax= 0.4, vmin = 0)
plot2= lambda x: plt.imshow(x,cmap=plt.cm.hot, vmax= 0.8, vmin = 0)

imgNum = 10
plt.clf()
plt.subplot(231)
a =  np.abs(tstLabel[imgNum,0,:,:])
plot1(a)
plt.axis('off')
plt.title('GT KT')
plt.subplot(232)
a =  np.abs(MapsUS[imgNum,0,:,:])
plot1(a)
plt.axis('off')
plt.title('US KT')
plt.subplot(233)
a =  np.abs(Kt[imgNum][0,0,:,:])
plot1(a)
plt.axis('off')
plt.title('Recon KT')




plt.subplot(234)
a =  np.abs(tstLabel[imgNum,1,:,:])
plot2(a)
plt.axis('off')
plt.title('GT Vp')
plt.subplot(235)
a =  np.abs(MapsUS[imgNum,1,:,:])
plot2(a)
plt.axis('off')
plt.title('US Vp')

plt.subplot(236)
a =  np.abs(Vp[imgNum][0,0,:,:])
plot2(a)
plt.axis('off')
plt.title('Recon Vp')


plt.axis('off')


plt.show()
#%%

temp_Kt = np.asarray(Kt)
temp_Vp = np.asarray(Vp)

temp = np.squeeze(np.concatenate((Kt,Vp),axis = 1))
#%%

save_dir = 'save_recon/'
save_file_name = save_dir + 'VTDCE_2_R20_PATB' +'.mat'
sio.savemat(save_file_name,{'GT':np.asarray(tstLabel),'MapUS':np.asarray( np.abs(MapsUS)),'recon':np.asarray( np.abs(temp))})
