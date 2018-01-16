import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable,grad
import torch
import torch.legacy.optim as legacyOptim
import torch.optim as optim
from torch.nn.modules.batchnorm import _BatchNorm
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
import torchvision.models as models

import time
from scipy.optimize import minimize_scalar
import weakref
from pdb import set_trace as bb
import collections
import numpy as np
import random
import uuid
from maze_env import MazeEnv
import re

maze="hard_maze.txt"
do_breadcrumb = True
breadcrumb = np.load(maze+".npy")

do_cuda = False 
if not do_cuda:
    torch.backends.cudnn.enabled = False

#global environment
env = None
max_episode_length = 600

#global archive to keep track of states sampled from environment
state_archive = collections.deque([], 200)

def set_maze(config):
    mazename = maze
    global env
    env = MazeEnv(mazename)

from pytorch_helpers import *

#default feed-forward net architecture
class netmodel(torch.nn.Module):
    def __init__(self, num_inputs, action_space,settings={}):
        super(netmodel, self).__init__()

        #should hidden units have biases?
        _b = True

        _init = 'xavier'

        num_outputs = action_space

        #overwritable defaults
        sz = 8  #how many neurons per layer
        hid_layers = 6 #how many hidden layers
        af = nn.ReLU 
        oaf = nn.Sigmoid

        afunc = {'relu':nn.ReLU,'tanh':nn.Tanh,'linear':lambda : linear,'sigmoid':nn.Sigmoid,'selu':selu,'silu':silu,'lrelu':nn.LeakyReLU}

        if 'size' in settings:
            sz = settings['size']
        if 'layers' in settings:
            hid_layers = settings['layers']
        if 'oaf' in settings:
            oaf = afunc[settings['oaf']] 
        if 'af' in settings:
            af = afunc[settings['af']]
        if 'init' in settings:
            _init = settings['init']

        self._af = af
        self.af = af()
        self.sigmoid = oaf()

        #first fully-connected layer changing from input-size representation to hidden-size representation
        self.fc1 = nn.Linear(num_inputs, sz, bias=_b) 

        self.hidden_layers = []

        #create all the hidden layers
        for x in range(hid_layers):
            self.hidden_layers.append(nn.Linear(sz, sz, bias=True))
            self.add_module("hl%d"%x,self.hidden_layers[-1])

        #create the hidden -> output layer
        self.fc_out = nn.Linear(sz, num_outputs, bias=_b)

        #initialize all the weights according to xavier init
        if _init=='xavier':
            self.apply(weight_init_xavier)

        self.train()

    def forward(self, inputs,intermediate=None,debug=False):

        x = inputs
        #fully connection input -> hidden
        x = self.fc1(x)
        x = self._af()(x)
        
        #propagate signal through hidden layers
        for idx,layer in enumerate(self.hidden_layers):

            #do fc computation
            x = layer(x)

            #run it through activation function
            x = self._af()(x)

            if intermediate == idx:
                cached = x
        
        #output layer
        x = self.fc_out(x)
        x = self.sigmoid(x)

        if intermediate!=None:
            return x,cached

        return x

    #function to return current pytorch gradient in same order as genome's flat vector theta
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

    #function to take a flat vector and reshape it to resemble neural network weights
    def reshape_parameters(self,pvec):
        count = 0
        numpy_params = []

        for name,param in self.named_parameters():
            sz = param.data.numpy().flatten().shape[0]
            raw = pvec[count:count + sz]
            reshaped = raw.reshape(param.data.numpy().shape)
            numpy_params.append((name,reshaped))
            count += sz

        #print ([ (r[0],(r[1]**2).sum().mean()) for r in numpy_params])
        return numpy_params

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


#genome class that can be mutated, selected, evaluated in domain
#substrate for evolution
class individual:
    env = None  #perhaps turn this into an env_generator? or pass into constructor? for parallelization..
    model_generator = None
    global_model = None
    rollout = None
    instances = []

    def __init__(self):
        self.noise = 0.05
        self.smog = False
        self.id = uuid.uuid4().int
        self.live_descendants = 0
        self.alive = True
        self.dead_weight = False
        self.parent= None
        self.percolate = False
        self.selected = 0

        if self.percolate:
            self.__class__.instances.append(weakref.proxy(self))

    def copy(self,percolate=False): 
        new_ind = individual()
        new_ind.genome = self.genome.copy()
        new_ind.states = self.states

        if self.percolate:
            new_ind.parent = self

        #update live descendant count
        if self.percolate:
            self.live_descendants += 1
            if hasattr(self,'parent'):
                p_pointer = self.parent
            else:
                p_pointer = None
                self.parent = None
            while p_pointer != None:
                p_pointer.live_descendants += 1 
                p_pointer = p_pointer.parent
        
        return new_ind

    def kill(self):
        self.alive=False
        if self.live_descendants <= 0:
                self.dead_weight=True
        self.remove_live_descendant()

    def remove_live_descendant(self):
        p_pointer = self.parent
        while p_pointer != None:
            p_pointer.live_descendants -= 1
            if p_pointer.live_descendants <= 0 and not p_pointer.alive:
                p_pointer.dead_weight=True

            p_pointer = p_pointer.parent

    def mutate(self, mutation='regular', **kwargs):

        #plain mutation is normal ES-style mutation
        if mutation=='regular':
            self.genome = mutate_plain(self.genome, states=self.states,**kwargs)
        elif mutation.count("SM-G")>0:
            #smog_target is target-based smog where we aim to perturb outputs
            self.genome = mutate_sm_g(
                    mutation,
                    self.genome,
                    individual.global_model,
                    individual.env,
                    states=self.states,
                    **kwargs)
            #smog_grad is TRPO-based smog where we attempt to induce limited policy change
        elif mutation.count("SM-R")>0:
                self.genome = mutate_sm_r(
                    self.genome,
                    individual.global_model,
                    individual.env,
                    states=self.states,
                    **kwargs)
        else:
            assert False

    #randomly initialize genome using underlying ANN's random init
    def init_rand(self):
        global controller_settings
        model = individual.model_generator
        env = individual.env
        newmodel = model(env.observation_space,env.action_space,controller_settings)
        self.genome = newmodel.extract_parameters()

    def render(self, screen):
        individual.global_model.inject_parameters(self.genome)
        reward, state_buffer, _beh = individual.rollout(
            {},
            individual.global_model,
            individual.env,
            render=True,
            screen=screen)

    #evaluate genome in environment with a roll-out
    def map(self, push_all=True, trace=False):
        global state_archive
        individual.global_model.inject_parameters(self.genome)
        reward, state_buffer, _beh, _behtrace,_broken = individual.rollout(
            {}, individual.global_model, individual.env, trace=trace)

        if push_all:
            state_archive = state_buffer
            self.states = state_buffer
        else:
            print("not using all states")
            state_archive.appendleft(random.choice(state_buffer))
            self.states = None

        self.broken = _broken
        self.behavior_trace = _behtrace

        self.reward = reward
        self.behavior = np.array(_beh)
        _breadcrumb_fitness(self)

        self.solved = individual.env.e.reachgoal

    #does individual solve the task?
    def solution(self):
        return self.solved

    #save genome out
    def save(self, fname):
        if fname.count(".npy")==0:
            fname_new=fname+".npy"
        else:
            fname_new=fname
        np.save(fname_new, self.genome)

    #load genome in
    def load(self, fname):
        if fname.count(".npy")==0:
            fname_new=fname+".npy"
        else:
            fname_new=fname
        self.genome = np.load(fname_new)
        print self.genome.shape
        print model.extract_parameters().shape 

#Method to conduct maze rollout
@staticmethod
def do_rollout(args, model, env, render=False, screen=None, trace=False):
    state_buffer = collections.deque([], 400)

    action_repeat = 3

    state = env._reset()
    state_buffer.appendleft(state)
    state = torch.from_numpy(state)
    this_model_return = 0

    beh_trace = []
    broken = False
    action = None

    # Rollout
    for step in range(max_episode_length):
        state = state.float()

        if True:  #(random.random()<0.05):
            state_buffer.appendleft(state.numpy())
        if trace:
            beh_trace.append(env._trace())

        if step%action_repeat==0:
         state = state.view(1, env.observation_space)
         logit = model(Variable(state, volatile=True))
         action = logit.data.numpy()[0]

         #if network is broken
         if np.isnan(action).any():
             broken=True
             break

        if render:
            env._render(screen)
            pygame.image.save(screen, "time%03d.png"%step)
            time.sleep(0.01)

        state, reward, done, _ = env._step(action)
        this_model_return += reward

        if done:
            break
        state = torch.from_numpy(state)

    beh = env._behavior()

    if broken==True:
        print "broken..."
        beh = np.array([0,0])

    return this_model_return, state_buffer, beh, beh_trace,broken


def mutate_plain(params, mag=0.05,**kwargs):
    do_policy_check = False

    delta = np.random.randn(*params.shape).astype(np.float32)*np.array(mag).astype(np.float32)
    new_params = params + delta

    diff = np.sqrt(((new_params - params)**2).sum())

    if do_policy_check:
        output_dist = check_policy_change(params,new_params,kwargs['states'])
        print("mutation size: ", diff, "output distribution change:",output_dist)
    else:
        print("mutation size: ", diff)

    return new_params


def mutate_sm_r(params,
                     model,
                     env,
                     verbose=True,
                     states=None,
                     mag=0.01):
    global state_archive

    model.inject_parameters(params.copy())

    if states == None:
        states = state_archive

    delta = np.random.randn(*(params.shape)).astype(np.float32)
    delta = delta / np.sqrt((delta**2).sum())

    sz = min(100,len(state_archive))
    np_obs = np.array(random.sample(state_archive, sz), dtype=np.float32)
    verification_states = Variable(
        torch.from_numpy(np_obs), requires_grad=False)

    output = model(verification_states)
    old_policy = output.data.numpy()

    #do line search
    threshold = mag
    search_rounds = 15
    
    def search_error(x,raw=False):
        new_params = params + delta * x
        model.inject_parameters(new_params)

        output = model(verification_states).data.numpy()

        change = ((output - old_policy)**2).mean()
        if raw:
            return change
        return (change-threshold)**2
    
    mult = minimize_scalar(search_error,tol=0.01**2,options={'maxiter':search_rounds,'disp':True})
    new_params = params+delta*mult.x

    print 'Distribution shift:',search_error(mult.x,raw=True)
    print "SM-R scaling factor:",mult.x
    diff = np.sqrt(((new_params - params)**2).sum())
    print("mutation size: ", diff)
    return new_params

def check_policy_change(p1,p2,states):
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


def mutate_sm_g(mutation, 
                       params,
                       model,
                       env,
                       verbose=False,
                       states=None,
                       mag=0.1,
					 	**kwargs):

    global state_archive

    #inject parameters into current model
    model.inject_parameters(params.copy())

    #if no states passed in, use global state archive
    if states == None:
        states = state_archive

    #sub-sample experiences from parent
    sz = min(100,len(states))
    verification_states = np.array(random.sample(states, sz), dtype=np.float32)
    verification_states = Variable(torch.from_numpy(verification_states), requires_grad=False)

    #run experiences through model
    #NOTE: for efficiency, could cache these during actual evalution instead of recalculating
    old_policy = model(verification_states)

    num_outputs = old_policy.size()[1]

    abs_gradient=False 
    avg_over_time=False
    second_order=False

    if mutation.count("ABS")>0:
        abs_gradient=True
        avg_over_time=True
    if mutation.count("SO")>0:
        second_order=True

    #generate normally-distributed perturbation
    delta = np.random.randn(*params.shape).astype(np.float32)*mag

    if second_order:
        print 'SM-G-SO'
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
        print "SM-G-SUM"
        tot_size = model.count_parameters()
        jacobian = torch.zeros(num_outputs, tot_size)
        grad_output = torch.zeros(*old_policy.size())

        for i in range(num_outputs):
            model.zero_grad()	
            grad_output.zero_()
            grad_output[:, i] = 1.0
            old_policy.backward(grad_output, retain_variables=True)
            jacobian[i] = torch.from_numpy(model.extract_grad())
  
        scaling = torch.sqrt(  (jacobian**2).sum(0) )

    else:
        print "SM-G-ABS"
        #NOTE: Expensive because quantity doesn't slot naturally into TF/pytorch framework
        tot_size = model.count_parameters()
        jacobian = torch.zeros(num_outputs, tot_size, sz)
        grad_output = torch.zeros([1,num_outputs]) #*old_policy.size())

        for i in range(num_outputs):
            for j in range(sz):
                old_policy_j = model(verification_states[j:j+1])
                model.zero_grad() 	
                grad_output.zero_()

                grad_output[0, i] = 1.0

                old_policy_j.backward(grad_output, retain_variables=True)
                jacobian[i,:,j] = torch.from_numpy(model.extract_grad())

        mean_abs_jacobian = torch.abs(jacobian).mean(2)
        scaling = torch.sqrt( (mean_abs_jacobian**2).sum(0))

    scaling = scaling.numpy()
   
    #Avoid divide by zero error 
    #(intuition: don't change parameter if it doesn't matter)
    scaling[scaling==0]=1.0

    #Avoid straying too far from first-order approx 
    #(intuition: don't let scaling factor become too enormous)
    scaling[scaling<0.01]=0.01

    #rescale perturbation on a per-weight basis
    delta /= scaling

    #generate new perturbation
    new_params = params+delta

    model.inject_parameters(new_params)
    old_policy = old_policy.data.numpy()

    #restrict how far any dimension can vary in one mutational step
    weight_clip = 0.2

    #currently unused: SM-G-*+R (using linesearch to fine-tune)
    mult = 0.05

    if mutation.count("R")>0:
        linesearch=True
        threshold = mag
    else:
        linesearch=False

    if linesearch == False:
        search_rounds = 0
    else:
        search_rounds = 15

    def search_error(x,raw=False):
        final_delta = delta*x
        final_delta = np.clip(final_delta,-weight_clip,weight_clip)
        new_params = params + final_delta
        model.inject_parameters(new_params)

        output = model(verification_states).data.numpy()
        change = np.sqrt(((output - old_policy)**2).sum(1)).mean()

        if raw:
            return change

        return (change-threshold)**2

    if linesearch:
        mult = minimize_scalar(search_error,bounds=(0,0.1,3),tol=(threshold/4)**2,options={'maxiter':search_rounds,'disp':True})
        print "linesearch result:",mult
        chg_amt = mult.x
    else:
        #if not doing linesearch
        #don't change perturbation
        chg_amt = 1.0

    final_delta = delta*chg_amt
    print 'perturbation max magnitude:',final_delta.max()

    final_delta = np.clip(delta,-weight_clip,weight_clip)
    new_params = params + final_delta

    print 'max post-perturbation weight magnitude:',abs(new_params).max()

    if verbose:
        print("divergence:", search_error(chg_amt,raw=True))

    diff = np.sqrt(((new_params - params)**2).sum())
    print("mutation size: ", diff)

    return new_params


model = None
controller_settings = None

def setup(maze,_controller_settings):
    global model,controller_settings
    set_maze(maze)
    controller_settings = _controller_settings
    model = netmodel(env.observation_space, env.action_space,controller_settings)

    if do_cuda:
        model.cuda()

    individual.env = env
    individual.model_generator = netmodel
    individual.rollout = do_rollout
    individual.global_model = model

#breadcrumb finess calculation (given breadcrumb array)
def _breadcrumb_fitness(ind):
    global breadcrumb
    pos_x = int(ind.behavior[-2])
    pos_y = int(ind.behavior[-1])
    ind.fitness = -breadcrumb[pos_x,pos_y]
    if ind.broken:
     ind.fitness = -1e8


if __name__ == '__main__':
    import pygame
    from pygame.locals import *
    pygame.init()

    pygame.display.set_caption('Viz')
    SZX=SZY=400
    screen = pygame.display.set_mode((SZX, SZY))

    solution_file="solution.npy"

    setup({'maze':"hard_maze.txt"},{'layers':64,'af':'tanh','size':125,'residual':True})
    robot = individual()
    robot.load(solution_file)
    robot.render(screen)

