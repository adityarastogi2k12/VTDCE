import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class vt_net(nn.Module):
    def __init__(self):
        super(vt_net, self).__init__()
# (inp channels, Out Channels, Kernel Size, Stride, Padding)
        
        # Down 1
        
        self.conv10 = nn.Sequential(nn.Conv3d(1,32,3,1,1),nn.BatchNorm3d(32),
        nn.ReLU())
        
        self.conv11 = nn.Sequential(nn.Conv3d(32,32,3,1,1),nn.BatchNorm3d(32),
        nn.ReLU())
        
        self.down1  = nn.MaxPool3d(2,2)
        
        # Down 2
        
        self.conv20 = nn.Sequential(nn.Conv3d(32,64,3,1,1),nn.BatchNorm3d(64),
        nn.ReLU())
        
        self.conv21 = nn.Sequential(nn.Conv3d(64,64,3,1,1),nn.BatchNorm3d(64),
        nn.ReLU())
        
        self.down2  = nn.MaxPool3d(2,2)

        # Down 3
        
        self.conv30 = nn.Sequential(nn.Conv3d(64,128,3,1,1),nn.BatchNorm3d(128),
        nn.ReLU())
        
        self.conv31 = nn.Sequential(nn.Conv3d(128,128,3,1,1),nn.BatchNorm3d(128),
        nn.ReLU())
        
        self.down3  = nn.MaxPool3d(2,2)

         # Down 4
        
#        self.conv40 = nn.Sequential(nn.Conv3d(128,256,3,1,1),nn.BatchNorm3d(256),
#        nn.ReLU())
#        
#        self.conv41 = nn.Sequential(nn.Conv3d(256,256,3,1,1),nn.BatchNorm3d(256),
#        nn.ReLU())
#        
#        self.down4  = nn.MaxPool3d(2,2)

        # Middle

        self.convm1 = nn.Sequential(nn.Conv2d(128,256,3,1,1),nn.BatchNorm2d(256),
        nn.ReLU())
        
        self.convm2 = nn.Sequential(nn.Conv2d(256,256,3,1,1),nn.BatchNorm2d(256),
        nn.ReLU())

        # Up 1

        self.up1   = nn.ConvTranspose2d(256,128,2,2)
        
        self.conv50 = nn.Sequential(nn.Conv2d(256,128,3,1,1),nn.BatchNorm2d(128),
        nn.ReLU(inplace = True))
        
        self.conv51 = nn.Sequential(nn.Conv2d(128,128,3,1,1),nn.BatchNorm2d(128),
        nn.ReLU(inplace = True))

        # Up 2

        self.up2   = nn.ConvTranspose2d(128,64,2,2)

        self.conv60 = nn.Sequential(nn.Conv2d(128,64,3,1,1),nn.BatchNorm2d(64),
        nn.ReLU(inplace = True))
        self.conv61 = nn.Sequential(nn.Conv2d(64,64,3,1,1),nn.BatchNorm2d(64),
        nn.ReLU(inplace = True))

        # Up 3

        self.up3   = nn.ConvTranspose2d(64,32,2,2)

        self.conv70 = nn.Sequential(nn.Conv2d(64,32,3,1,1),nn.BatchNorm2d(32),
        nn.ReLU(inplace = True))
        
        self.conv71 = nn.Sequential(nn.Conv2d(32,32,3,1,1),nn.BatchNorm2d(32),
        nn.ReLU(inplace = True))

        # Up 4

#        self.up4   = nn.ConvTranspose2d(64,32,2,2)
#
#        self.conv80 = nn.Sequential(nn.Conv2d(64,32,3,1,1),nn.BatchNorm2d(32),
#        nn.ReLU(inplace = True))
#        
#        self.conv81 = nn.Sequential(nn.Conv2d(32,32,3,1,1),nn.BatchNorm2d(32),
#        nn.ReLU(inplace = True))
#        
      

        self.out1 = nn.Conv2d(32,1,1,1)
        
        self.conv80 = nn.Sequential(nn.Conv2d(64,32,3,1,1),nn.BatchNorm2d(32),
        nn.ReLU(inplace = True))
        
        self.conv81 = nn.Sequential(nn.Conv2d(32,32,3,1,1),nn.BatchNorm2d(32),
        nn.ReLU(inplace = True))
        
      

        self.out2 = nn.Conv2d(32,1,1,1)
        self.skip = nn.Identity()
        self._init_weights()
        
    def _init_weights(self):        
         for m in self.modules():             
            if isinstance(m, nn.Conv3d):
               torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
               m.bias.data.fill_(0.01)   
            elif isinstance(m, nn.Conv2d):
               torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
               m.bias.data.fill_(0.01)
               
    def forward(self, x):
        
        x10 = self.conv10(x)
       	x11 = self.conv11(x10)
       	x_d1 = self.down1(x11)
        
       	x20 = self.conv20(x_d1)
       	x21 = self.conv21(x20)
       	x_d2 = self.down2(x21)        
        
       	x30 = self.conv30(x_d2)
       	x31 = self.conv31(x30)
       	x_d3 = self.down3(x31)        
               
#       	x40 = self.conv40(x_d3)
#       	x41 = self.conv41(x40)
#       	x_d4 = self.down4(x41)
#        
        x_d3 = torch.squeeze(torch.mean(x_d3,2),2)
       	xm1 = self.convm1(x_d3)
        	
        x_u1 = self.up1(self.convm2(xm1))
            	
       	#x_u1 = self.up1(xm1)        
        
        x_c1 = torch.cat((torch.squeeze(torch.mean(x31,2),2) ,x_u1), dim = 1)
        #print(x_c1.size())
       	x50 = self.conv50(x_c1)
       	x51 = self.conv51(x50)
       	x_u2 = self.up2(x51)        
       
        x_c2 = torch.cat((torch.squeeze(torch.mean(x21,2),2) ,x_u2), dim = 1)
       	
        x60 = self.conv60(x_c2)
       	x61 = self.conv61(x60)
       	x_u3 = self.up3(x61)
        
        x_c3 = torch.cat((torch.squeeze(torch.mean(x11,2),2) ,x_u3), dim = 1)
       	x70 = self.conv70(x_c3)
       	x71 = self.conv71(x70)
       
        y1 = self.out1(x71)
        
#       	x_u4 = self.up4(x71)
           
#        x_c4 = torch.cat((torch.squeeze(torch.mean(x11,2),2) ,x_u4), dim = 1)
        #print(x_c4.size())
#       	x80 = self.conv80(x_c4)
#        x81 = self.conv81(x80)
#       
#        y1 = self.out1(x81)
#        
        x80 = self.conv80(x_c3)
        x81 = self.conv81(x80)
       
        y2 = self.out2(x81)
        #print(y2.size())
        #y = torch.cat((y1,y2),dim=1)
        return y1,y2
#%%


def edge_loss(output,label,if_cuda,gpuid):
  x_filter =torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])
  y_filter = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
  if if_cuda == True:
      x_filter = x_filter.cuda(gpuid)
      y_filter = y_filter.cuda(gpuid)
  label = label.unsqueeze(1)
#  print(label.size())
  x_weights = x_filter.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
  y_weights = y_filter.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
#  print(x_weights.size())
  g1_x = F.conv2d(output,x_weights,padding = 1)
  g1_y = F.conv2d(output,y_weights,padding = 1)


  g2_x = F.conv2d(label,x_weights,padding = 1)
  g2_y = F.conv2d(label,y_weights,padding = 1)
 #convx = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
#  print(g2_x.size())

  g_1 = torch.sqrt(torch.pow(g1_x, 2) + torch.pow(g1_y, 2))
  g_2 = torch.sqrt(torch.pow(g2_x, 2) + torch.pow(g2_y, 2))

  return torch.mean((g_1 - g_2).pow(2))



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#%%
#x = torch.rand(2,1,32,320,320) #[bs,in_ch,depth, H,W]
#net = vt_net()
##
##y1,y2 = net(x)
##print(y2.size())
#
#x = torch.rand(2,1,320,320) #[bs,in_ch,H,W]
#print(x.size())
#edge_loss(x,x[:,0,:,:])
#y = x[:,0,:,:].unsqueeze(1)
#print(x[:,0,:,:].size())
#print(y.size())
#
#
#p = count_parameters(net)
#print(p)
