import time
from scipy.optimize import minimize_scalar
import weakref
from pdb import set_trace as bb
import collections
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable,grad
import torch
import torch.legacy.optim as legacyOptim
import torch.optim as optim
from torch.nn.modules.batchnorm import _BatchNorm
from torch.autograd.gradcheck import zero_gradients

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import torchvision.models as models
import argparse

#toy model specified in pytorch
class netmodel(torch.nn.Module):
    def __init__(self):
        super(netmodel, self).__init__()
        self.w0 = Parameter(torch.Tensor(1))
        self.w1 = Parameter(torch.Tensor(1))
        #init params uniformly
        self.w0.data.uniform_(-1,1)
        self.w1.data.uniform_(-1,1)

    #model with two weights and two outputs
    def forward(self, inputs):
        x = inputs
        y = torch.stack([100*self.w0*inputs[:,0],0.1*self.w1*inputs[:,1]])
        y = torch.t(y)
        return y.contiguous()

    #function to return current pytorch gradient in same order as genome's flattened parameter vector
    def extract_grad(self):
        tot_size = self.count_parameters()
        pvec = np.zeros(tot_size, np.float32)
        count = 0
        for param in self.parameters():
            sz = param.grad.data.numpy().flatten().shape[0]
            pvec[count:count + sz] = param.grad.data.numpy().flatten()
            count += sz
        return pvec.copy()

    #function to grab current flattened neural network weights
    def extract_parameters(self):
        tot_size = self.count_parameters()
        pvec = np.zeros(tot_size, np.float32)
        count = 0
        for param in self.parameters():
            sz = param.data.numpy().flatten().shape[0]
            pvec[count:count + sz] = param.data.numpy().flatten()
            count += sz
        return pvec.copy()

    #function to inject a flat vector of ANN parameters into the model's current neural network weights
    def inject_parameters(self, pvec):
        tot_size = self.count_parameters()
        count = 0

        for param in self.parameters():
            sz = param.data.numpy().flatten().shape[0]
            raw = pvec[count:count + sz]
            reshaped = raw.reshape(param.data.numpy().shape)
            param.data = torch.from_numpy(reshaped)
            count += sz

        return pvec

    #count how many parameters are in the model
    def count_parameters(self):
        count = 0
        for param in self.parameters():
            #print param.data.numpy().shape
            count += param.data.numpy().flatten().shape[0]
        return count

#eventual X,Y pairs for training
states = None
targets = None



#model evaluation code (i.e. to calculate fitness)
def evaluate(model,param):

    #inject new parameters into model
    model.inject_parameters(param)

    #move from numpy into pytorch
    inputs = Variable(torch.from_numpy(states),requires_grad=False)
    y_target = Variable(torch.from_numpy(targets),requires_grad=False)

    #run inputs through model
    y=model(inputs)

    #calculate squared error
    error = (y-y_target)**2

    return - error.sum().data.numpy()[0]

#check how much outputs of network have changed from parameter settings p1 changing to p2
def check_policy_change(p1,p2,model,states):
    model.inject_parameters(p1.copy())
    #TODO: check impact of greater accuracy
    sz = min(100,len(states))

    verification_states = np.array(random.sample(states, sz), dtype=np.float32)
    verification_states = Variable(torch.from_numpy(verification_states), requires_grad=False)
    old_policy = model(verification_states).data.numpy()
    old_policy = Variable(torch.from_numpy(old_policy), requires_grad=False)

    model.inject_parameters(p2.copy())
    model.zero_grad()
    new_policy = model(verification_states)
    divergence_loss_fn = torch.nn.MSELoss(size_average=True)
    divergence_loss = divergence_loss_fn(new_policy,old_policy)

    return divergence_loss.data[0]

#vanilla gaussian perturbation mutation
def mutate_plain(mutation,params, mag=0.05,**kwargs):
    noise=mag
    verbose = False
    do_policy_check = True

    #create gaussian perturbation
    delta = np.random.randn(*params.shape).astype(np.float32)*np.array(noise).astype(np.float32)
    #add to parameter vector
    new_params = params + delta

    diff = (abs(new_params - params)).sum()

    if do_policy_check:
        output_dist = check_policy_change(params,new_params,model=kwargs['model'],states=kwargs['states'])
        if verbose:
            print("diff: ", diff, "od:",output_dist)
    else:
        if verbose:
            print("diff: ", diff)

    return new_params.copy(), delta

#SM implementation
def mutate_sm(mutation,params,
                       model=None,
                       env=None,
                       verbose=False,
                       states=None,
                       mag=0.1,
					 	**kwargs):

    model.inject_parameters(params.copy())

    #TODO: why?
    _states = np.concatenate((states,states,states,states))

    #grab old policy
    sz = min(100,len(_states))

    #experience in this domain = the classification *input* patterns  
    experience_states = _states
    experience_states = Variable(torch.from_numpy(experience_states), requires_grad=False)

    old_policy = model(experience_states)
    num_classes = old_policy.size()[1]

    #SM-ABS
    abs_gradient=False 

    #SM-SO
    second_order=False

    #SM-R
    sm_r = False

    #SM-R uses a line search 
    linesearch=False


    if mutation.count("SM-R")>0:
        sm_r = True
    elif mutation.count("SO")>0:
        second_order=True
    elif mutation.count("ABS")>0:
        abs_gradient=True

    #initial perturbation
    delta = np.random.randn(*params.shape).astype(np.float32)*mag

    if sm_r:
        #print "SM-R"
        scaling = torch.ones(params.shape)
        linesearch = True
    elif second_order:
        #print "SM-G-SO"
        np_copy = np.array(old_policy.data.numpy(),dtype=np.float32)
        _old_policy_cached = Variable(torch.from_numpy(np_copy), requires_grad=False)
        loss =  ((old_policy-_old_policy_cached)**2).sum(1).mean(0)
        loss_gradient = grad(loss, model.parameters(), create_graph=True)
        flat_gradient = torch.cat([grads.view(-1) for grads in loss_gradient]) #.sum()

        direction = (delta/ np.sqrt((delta**2).sum()))
        direction_t = Variable(torch.from_numpy(direction),requires_grad=False)
        grad_v_prod = (flat_gradient * direction_t).sum()
        second_deriv = torch.autograd.grad(grad_v_prod, model.parameters())
        sensitivity = torch.cat([g.contiguous().view(-1) for g in second_deriv])
        scaling = torch.sqrt(torch.abs(sensitivity).data)

    elif not abs_gradient:
        #print "SM-G-SUM"
        tot_size = model.count_parameters()
        jacobian = torch.zeros(num_classes, tot_size)
        grad_output = torch.zeros(*old_policy.size())

        for i in range(num_classes):
            model.zero_grad()
            grad_output.zero_()
            grad_output[:, i] = 1.0

            old_policy.backward(grad_output, retain_variables=True)
            jacobian[i] = torch.from_numpy(model.extract_grad())

        scaling = torch.sqrt(  (jacobian**2).sum(0) )
    else:
        #print "SM-G-ABS"
        tot_size = model.count_parameters()
        jacobian = torch.zeros(num_classes, tot_size, sz)
        grad_output = torch.zeros(*old_policy.size())

        for i in range(num_classes):
            for j in range(sz):
                old_policy_new = model(experience_states[j:j+1]) 
                model.zero_grad() 	
                grad_output.zero_()

                grad_output[:, i] = 1.0/sz

                old_policy_new.backward(grad_output, retain_variables=True)
                jacobian[i,:,j] = torch.from_numpy(model.extract_grad())

        mean_abs_jacobian = torch.abs(jacobian).mean(2)
        scaling = torch.sqrt( (mean_abs_jacobian**2).sum(0))

    scaling = scaling.numpy()

    if verbose:
        print 'scaling sum',scaling.sum()
    
    scaling[scaling==0]=1.0
    scaling[scaling<0.01]=0.01
    
    old_delta = delta.copy()
    delta /= scaling
    new_params = params+delta
    model.inject_parameters(new_params)

    threshold = mag
    weight_clip = 10.0 #note generally probably should be smaller
    search_rounds = 15
    old_policy = old_policy.data.numpy()

    def search_error(x,raw=False):
        final_delta = delta*x
        final_delta = np.clip(final_delta,-weight_clip,weight_clip)
        new_params = params + final_delta
        model.inject_parameters(new_params)

        output = model(experience_states).data.numpy()

        change = np.sqrt(((output - old_policy)**2).sum(1)).mean()

        if raw:
            return change

        return np.sqrt(change-threshold)**2

    if linesearch:
        mult = minimize_scalar(search_error,bounds=(0,0.1,3),tol=(threshold/4),options={'maxiter':search_rounds,'disp':True})
        new_params = params+delta*mult.x
        chg_amt = mult.x
    else:
        chg_amt = 1.0

    final_delta = delta*chg_amt
    final_delta = np.clip(final_delta,-weight_clip,weight_clip)  #as 1.0

    new_params = params + final_delta

    if verbose:
        print 'delta max:',final_delta.max()
        print("divergence:", check_policy_change(params,new_params,model,states))
        print(new_params.shape,params.shape)
    diff = np.sqrt(((new_params - params)**2).sum())
    if verbose:
        print("diff: ", diff)

    return new_params.copy(),final_delta



#MAIN EXPERIMENTAL CODE
def main():
    global states,targets

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain",help="what incentive to drive search by",default="easy")
    parser.add_argument("--mutation", help="whether to use regular or SM mutations",default="regular")
    parser.add_argument("--mutation_mag", help="magnitude of mutation operator",default=0.01)
    args = parser.parse_args()


    #domain defenitions
    states0 = np.array([ [0,1],[0,1]],dtype=np.float32)
    targets0 = np.array( [[0,1],[0,1]],dtype=np.float32)

    states1 = np.array([ [1,0],[0,1]],dtype=np.float32)
    targets1 = np.array( [[1,0],[0,1]],dtype=np.float32)

    states2 = np.array([ [1.0,1.0],[-1.0,-1.0]],dtype=np.float32)
    targets2 = np.array( [[1,1.0],[-1,-1.0]],dtype=np.float32)

    if args.domain == 'easy':
        states = states0
        targets = targets0

    if args.domain == 'medium':
        states = states1
        targets = targets1

    if args.domain == 'washout':
        states = states2
        targets = targets2


    m = netmodel()
    theta = m.extract_parameters()

    if args.mutation!='regular':
        mutate = mutate_sm
    else:
        mutate = mutate_plain

    fit = None
    magnitude = float(args.mutation_mag)
    it = 0

    perf = []

    #print table heading
    #(you can pipe experiments to .csv)
    print "Iteration, fitness, theta0, theta1"

    #hill-climbing loop
    while it<2000:

        if fit==None:
            m.inject_parameters(theta)
            fit = evaluate(m,theta)

        print '%d, %f, %f, %f' % (it,fit,theta[0],theta[1]) 
        theta_prime,perturb = mutate(args.mutation,theta,model=m,states=states,mag=magnitude)
        fit_prime = evaluate(m,theta_prime)

        if fit_prime > fit:
            theta = theta_prime
            fit = fit_prime
        perf.append(fit)
        it+=1

if __name__=='__main__':
    main()
