import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import time
import math
import numpy as np
from pdb import set_trace as bb

n_hidden = 20
n_epochs = 10000
print_every = 100
plot_every = 1000
learning_rate = 1e-4 # If you set this too high, it might explode. If too low, it might not learn



def make_bitwise(num_bits,func):
 """
 Data generator for bit-wise tasks
 
 Arguments:
 num_bits -- the length of bit-sequences to generate
 func -- a python function that transforms a bit-sequence to an output
 """
 prelude = 0
 max_length = prelude+num_bits
 examples = 2**num_bits
 input_size = 1

 sequences = np.zeros((max_length,examples,input_size),np.float32)
 lengths = np.zeros((examples,),np.int32)
 labels = np.zeros((examples,),np.int64)

 lengths[:] = max_length

 for num in xrange(examples):
     binary = "0"*prelude + "{0:0%db}"%num_bits
     binary = binary.format(num)
     binary = [int(k) for k in binary]
     sequences[:,num,0] = binary
     labels[num] = func(binary)

 return sequences,lengths,labels

#create recurrent parity task (bits of length 4)
sequences,lengths,labels = make_bitwise(4,lambda x:sum(x)%2)
state_archive = sequences,lengths 

def Batch():
    """
    format data into batch for pytorch
    """
    global labels,lengths,sequences
    tInputs = Variable(torch.from_numpy(sequences))
    tLabels = Variable(torch.from_numpy(labels))
    tLengths = Variable(torch.from_numpy(lengths))
    return tInputs,tLengths,tLabels

class recurrent_model(nn.Module):
    """
    Simple recurrent NN model
    """
    def __init__(self, input_size, hidden_size, output_size,model_type='rnn'):
        """
        Initializer

        Arguments:
        model_type -- cell type, can be rnn or lstm or gru
        """
        super(recurrent_model, self).__init__()

        self.hidden_size = hidden_size
        self.model_type = model_type

        if model_type=='lstm':
         self.rnn_layer = nn.LSTMCell(input_size,hidden_size)
         self.rnn_layer2 = nn.LSTMCell(hidden_size,hidden_size)
        elif model_type=='gru':
         self.rnn_layer = nn.GRUCell(input_size,hidden_size)
         self.rnn_layer2 = nn.GRUCell(hidden_size,hidden_size)
        else:
         self.rnn_layer = nn.RNNCell(input_size,hidden_size)
         self.rnn_layer2 = nn.RNNCell(hidden_size,hidden_size)

        self.output_layer = nn.Linear(hidden_size,output_size)
        self.log_softmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()

    def forward_selfcontained(self,input,log=False):
        """
        Run a full sequence through the NN
        """
        length = input.size()[0]
        hidden = self.initHidden(input.size()[1])

        for i in range(length):
            output, hidden = self(input[i], hidden, log)

        return output

    def forward(self, input, hidden, log=False):
        """
        Run one time step through the NN
        """
        hidden[0] = self.rnn_layer(input,hidden[0])

        if isinstance(hidden[0],tuple):
            hidden_o = hidden[0][0]
        else:
            hidden_o = hidden[0]

        hidden[1] = self.rnn_layer2(hidden_o,hidden[1])

        if isinstance(hidden[1],tuple):
            hidden_o = hidden[1][0]
        else:
            hidden_o = hidden[1]

        
        output = self.output_layer(hidden_o)

        if log:
            output = self.log_softmax(output)
        else:
            output = self.softmax(output)

        return output, hidden

    def initHidden(self,batch_size):
        """
        helper function to initialize rnn hidden state
        """
        hiddens = []

        lstm = self.model_type=='lstm'

        if lstm:
            h0 = Variable(torch.zeros(batch_size, self.hidden_size))
            c0 = Variable(torch.zeros(batch_size, self.hidden_size))
            hiddens.append( (h0,c0) )

            h1 = Variable(torch.zeros(batch_size, self.hidden_size))
            c1 = Variable(torch.zeros(batch_size, self.hidden_size))
            hiddens.append( (h1,c1) )
        else:
            hiddens.append( Variable(torch.zeros(batch_size, self.hidden_size) ) )
            hiddens.append( Variable(torch.zeros(batch_size, self.hidden_size) ) )

        return hiddens


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

        print ([ (r[0],(r[1]**2).sum().mean()) for r in numpy_params])
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


def test_model(rnn):
    """

    """
    correct=0.0
    total=0.0
    nll = 0.0

    c_idx = 0

    _inp,_len,_lab = Batch()
    output, loss = train(rnn,_inp,_len,_lab,opt=False)
    loss = criterion(output, _lab)

    _d = output.max(1)[1].data.numpy()
    _l = _lab.data.numpy()
    correct = (_d==_l).sum()

    return float(correct)/(_d).shape[0],loss.data[0]
    
criterion = nn.NLLLoss()

def train(rnn,_inp,_len,_lab,opt=True):
    """
    If you want to try with SGD
    """
    global criterion

    if opt:
     optimizer.zero_grad()

    hidden = rnn.initHidden(int(_len.size()[0]))
    for i in range(int(_len.data[0])):
        output, hidden = rnn(_inp[i], hidden,log=True)

    loss = criterion(output, _lab)

    if opt:
     loss.backward()
     optimizer.step()

    return output, loss.data[0]

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

#you can also train directly with sgd
if __name__=='__main__':
    start = time.time()
    predict = False 
    if predict:
        rnn = torch.load('out.pt')
        print test_model(rnn)
        print rnn.extract_parameters().shape
        exit()

    rnn = recurrent_model(1, n_hidden, 2)

    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate,momentum=0.9)
    #optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    #optimizer = torch.optim.RMSprop(rnn.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    current_loss = 0
    all_losses = []

    for epoch in range(1, n_epochs + 1):
        #category, line, category_tensor, line_tensor = randomTrainingPair()
        _inp,_len,_lab = Batch()
        output, loss = train(rnn,_inp,_len,_lab)
        current_loss += loss

		# Print epoch number, loss, name and guess
        if epoch % print_every == 0:
            guess = output
            correct = 0
            line =''

            print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, float(epoch) / n_epochs * 100, timeSince(start), loss, line, guess, correct))
            print test_model(rnn)

        # Add current loss avg to list of losses
        if epoch % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    torch.save(rnn, 'out.pt')

