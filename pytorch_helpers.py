import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from pdb import set_trace as bb

#default neat-style init
def weight_init_default(m):
    if isinstance(m, nn.Linear) or isinstance(m,nn.Conv2d):
        m.weight.data.uniform_(-1.0,1.0)

#xavier initialization (apparently better than pytorch default)
def weight_init_xavier(m): 
    #print "xavier init."
    if isinstance(m, nn.Linear) or isinstance(m,nn.Conv2d):
        size = m.weight.size()
        fan_out = size[0] # number of rows
        fan_in = np.prod(size[1:]) # number of columns
        variance = np.sqrt(2.0/(fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)

#xavier initialization (apparently better than pytorch default)
#now He initialization for conv layers
def weight_init_he(m): 
    #print "he init."
    if isinstance(m, nn.Linear) or isinstance(m,nn.Conv2d):
        #print "xavierizing..."
        size = m.weight.size()
        fan_out = size[0] # number of rows
        fan_in = np.prod(size[1:]) # number of columns
        #variance = np.sqrt(2.0/(fan_in + fan_out))
        variance = np.sqrt(2.0) * np.sqrt(1.0/ fan_in)
        m.weight.data.normal_(0.0, variance)
        #print fan_out,fan_in,variance

def weight_norm(m): 
    if isinstance(m, nn.Linear) or isinstance(m,nn.Conv2d):
        nn.utils.weight_norm(m)

#layer norm (control mechanism for dealing with really deep nets)
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1).expand_as(x)
        std = x.std(-1).expand_as(x)
     
        return self.gamma.expand_as(x) * (x - mean) / (std + self.eps) + self.beta.expand_as(x)

def linear(x):
    return x

class selu(nn.Module):
    def __init__(self):
        super(selu, self).__init__()
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946
    def forward(self, x):
        temp1 = self.scale * F.relu(x)
        temp2 = self.scale * self.alpha * (F.elu(-1*F.relu(-1*x)))
        return temp1 + temp2

class silu(nn.Module):
    def __init__(self):
        super(selu, self).__init__()
    def forward(self, x):
        return x*F.sigmoid(x)

