import torch.nn as nn, torch, copy, tqdm, math
from torch.autograd import Variable
import torch.nn.functional as F
import pickle
import numpy as np
use_cuda = torch.cuda.is_available()



# encode the hidden states of a number of utterances
class InterUtteranceEncoder(nn.Module):
    def __init__(self, hid_size, inp_size, options):
        super(InterUtteranceEncoder, self).__init__()
        self.hid_size = hid_size
        self.rnn = nn.GRU(hidden_size=hid_size, input_size=inp_size,
                          num_layers=1, bidirectional=False, batch_first=True, dropout=options.drp)

    def forward(self, x):
        h_0 = Variable(torch.zeros(1, x.size(0), self.hid_size))
        if use_cuda:
            h_0 = h_0.cuda()
        # output, h_n for output batch is already dim 0
        h_o, h_n = self.rnn(x, h_0)
        h_n = h_n.view(x.size(0), -1, self.hid_size)
        return h_n
