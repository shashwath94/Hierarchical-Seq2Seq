import torch.nn as nn, torch, copy, tqdm, math
from torch.autograd import Variable
import torch.nn.functional as F
import pickle
import numpy as np
use_cuda = torch.cuda.is_available()


from Utterance_encoder import *
from InterUtterance_encoder import *
from Decoder import *



class HSeq2seq(nn.Module):
    def __init__(self, options):
        super(HSeq2seq, self).__init__()
        self.seq2seq = options.seq2seq
        self.utt_enc = UtteranceEncoder(options.vocab_size, options.emb_size, options.ut_hid_size, options)
        self.intutt_enc = InterUtteranceEncoder(options.ses_hid_size, options.ut_hid_size, options)
        self.dec = Decoder(options)

    def forward(self, batch):
        u1, u1_lenghts, u2, u2_lenghts, u3, u3_lenghts = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
        if use_cuda:
            u1 = u1.cuda()
            u2 = u2.cuda()
            u3 = u3.cuda()

        if self.seq2seq:
            o1, o2 = self.utt_enc((u1, u1_lenghts)), self.utt_enc((u2, u2_lenghts))
            qu_seq = torch.cat((o1, o2), 2)
            #final_session_o = self.intutt_enc(qu_seq)
            preds, lmpreds = self.dec((qu_seq, u3, u3_lenghts))
        else:
            o1, o2 = self.utt_enc((u1, u1_lenghts)), self.utt_enc((u2, u2_lenghts))
            qu_seq = torch.cat((o1, o2), 1)
            final_session_o = self.intutt_enc(qu_seq)
            preds, lmpreds = self.dec((final_session_o, u3, u3_lenghts))

        return preds, lmpreds
