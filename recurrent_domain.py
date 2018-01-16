from scipy.optimize import minimize_scalar
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad
import random
import collections
import uuid
import torch
import functools
import recurrent
import numpy as np
import torch

controller_settings = None
domain_settings = None

state_archive = recurrent.state_archive

from pdb import set_trace as bb
do_cuda = False 
if not do_cuda:
    torch.backends.cudnn.enabled = False

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
        newmodel = model() #model(env.observation_space,env.action_space,controller_settings)
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
    def map(self, push_all=False, trace=False):
        global state_archive

        individual.global_model.inject_parameters(self.genome)

        reward, state_buffer, _beh, _behtrace,_broken = individual.rollout(
            domain_settings, individual.global_model, trace=trace)

        if push_all:
            state_archive = state_buffer
            self.states = state_buffer
        else:
            #print("not using all states")
            #state_archive.appendleft(random.choice(state_buffer))
            self.states = None
        self.broken = _broken
        self.behavior_trace = _behtrace
        self.reward = reward
        self.fitness = self.reward
        self.behavior = np.array(_beh)
        self.solved = self.behavior[0]==1.0 

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

xs =[]
ys =[]

def mutate_sm_g(mutation,params,
                       model,
                       env,
                       verbose=False,
                       states=None,
                       mag=0.1,
					 	**kwargs):

    #deal with abs gradient
    abs_gradient=False 

    global state_archive
    if states == None:
        states = state_archive

    model.inject_parameters(params.copy())

    #initial perturbation
    delta = np.random.randn(*params.shape).astype(np.float32)*mag

    #grab old policy
    sz = min(100,states[0].shape[1])
    verification_states = states[0][:,np.random.choice(states[0].shape[1],size=sz)] 
    verification_states = Variable(torch.from_numpy(verification_states), requires_grad=False)

    old_policy = model.forward_selfcontained(verification_states)
    num_outputs = old_policy.size()[1]

    abs_gradient=False 
    second_order=False

    if mutation.count("ABS")>0:
        abs_gradient=True
    if mutation.count("SO")>0:
        second_order=True
    if mutation.count("R")>0:
        linesearch=True
    else:
        linesearch=False

    if second_order:
        print 'SM-G-SO'
        np_copy = np.array(old_policy.data.numpy(),dtype=np.float32)
        _old_policy_cached = Variable(torch.from_numpy(np_copy), requires_grad=False)
        loss =  ((old_policy-_old_policy_cached)**2).sum(1).mean(0)
        loss_gradient = grad(loss, model.parameters(), create_graph=True)
        flat_gradient = torch.cat([grads.view(-1) for grads in loss_gradient]) #.sum()

        step = 1.0
        direction = (delta/ np.sqrt((delta**2).sum())) * step
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
        tot_size = model.count_parameters()
        jacobian = torch.zeros(num_outputs, tot_size, sz)
        grad_output = torch.zeros([1,num_outputs]) #*old_policy.size())

        for i in range(num_outputs):
            for j in range(sz):
                old_policy_j = model.forward_selfcontained(verification_states[:,j:j+1])
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

    threshold = mag 

    if linesearch == False:
        seach_rounds = 0
    else:
        search_rounds = 15

    def search_error(x,raw=False,**kwargs):
        new_params = params + delta * x
        model.inject_parameters(new_params)

        if 'policy' in kwargs:
            policy_o = kwargs['policy']
            del kwargs['policy']
        else:
            policy_o = old_policy

        output = model.forward_selfcontained(verification_states,**kwargs).data.numpy()

        
        change = ((output - policy_o)**2).sum(1).mean()

        if raw:
            return change

        return np.sqrt(change-threshold)**2

    if linesearch:
        old_policy = old_policy.data.numpy()
        mult = minimize_scalar(trpo_error,bounds=(0,0.1,3),tol=(threshold/4),options={'maxiter':trpo_rounds,'disp':True})
        new_params = params+delta*mult.x
        chg_amt = mult.x
    else:
        chg_amt = 1.0

    diff = np.sqrt(((new_params - params)**2).sum())
    chg_amt = 1.0

    print("diff: ", diff, "od:",search_error(chg_amt,raw=True,policy=old_policy))

    return new_params

def mutate_sm_r(params,
                     model,
                     env,
                     mag=0.01,
                     states=None,**kwargs
                     ):

    global state_archive

    model.inject_parameters(params.copy())

    if states == None or len(states)==0:
        states = state_archive

    delta = np.random.randn(*(params.shape)).astype(np.float32)
    delta = delta / np.sqrt((delta**2).sum())

    #independent sampling..
    sz = min(100,len(states))
    c_obs = states[0][:,np.random.choice(states[0].shape[1],size=sz)] 

    observations = Variable(torch.from_numpy(c_obs), requires_grad=False)

    output = model.forward_selfcontained(observations)
    old_policy = output.data.numpy()

    threshold = mag

    search_rounds = 15
    verification_states = observations

    def search_error(x,raw=False):
        new_params = params + delta * x
        model.inject_parameters(new_params)

        output = model.forward_selfcontained(verification_states).data.numpy()
        #output = np.clip(output.data.numpy(),0,1)

        change = ((output - old_policy)**2).mean()
        if raw:
            return change
        return (change-threshold)**2
    
    mult = minimize_scalar(search_error,tol=0.01**2,options={'maxiter':search_rounds,'disp':True})
    new_params = params+delta*mult.x

    print "SM-R scaling factor:",mult.x

    diff = np.sqrt(((new_params - params)**2).sum())

    print("mutation magnitude: ", diff, "divergence:",search_error(mult.x,raw=True))
    return new_params

def check_policy_change(p1,p2,states):
    model.inject_parameters(p1.copy())
    #TODO: check impact of greater accuracy
    sz = min(100,len(states))

    verification_states = states[np.random.choice(states.shape[0],size=sz)] #np.array(random.sample(states, sz), dtype=np.float32)
    verification_states = Variable(torch.from_numpy(verification_states), requires_grad=False)
    old_policy = model.forward_selfcontained(verification_states).data.numpy()
    old_policy = Variable(torch.from_numpy(old_policy), requires_grad=False)

    model.inject_parameters(p2.copy())
    model.zero_grad()
    new_policy = model.forward_selfcontained(verification_states)
    divergence_loss_fn = torch.nn.MSELoss(size_average=True)
    divergence_loss = divergence_loss_fn(new_policy,old_policy)

    return divergence_loss.data[0]


@staticmethod
def do_rollout(args, model, trace=False):
    state_buffer = collections.deque([], 400)

    correct, this_model_return = recurrent.test_model(model)

    if args['incentive']=='fitness':
        this_model_return = -this_model_return
    else:
        this_model_return = correct

    print "Correct:",correct, ' return:', -this_model_return

    broken = False
    beh_trace = None
    beh = np.array([correct])

    return this_model_return, state_buffer, beh, beh_trace,broken


def setup(_domain_settings,_controller_settings):
    global model,controller_settings,domain_settings
    controller_settings = _controller_settings
    inp_size = 1
    hid_size = controller_settings['size']
    out_size = 2
    model = controller_settings['af']

    model_generator = functools.partial(recurrent.recurrent_model,input_size=inp_size,hidden_size=hid_size,output_size=out_size,model_type=model)

    model = model_generator() #rnn.RNN(env.observation_space, env.action_space,controller_settings)
    
    if do_cuda:
        model.cuda()
    
    domain_settings = _domain_settings

    individual.model_generator = model_generator
    individual.rollout = do_rollout
    individual.global_model = model


if __name__ == '__main__':

    setup({'incentive':'fitness'},{'af':'sigmoid','size':20,'residual':False,'incentive':'fitness'})
    robot = individual()
    robot.load("solution.npy")
    robot.map()
    print robot.behavior
    print robot.reward


