import torch
import copy
import pickle
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np
from tqdm import tqdm
from collections import Counter

use_cuda = torch.cuda.is_available()

'''
Function to batchify the dataset for faster processing. To make sure each batch has a
sequence length of the longest sentence in that batch.
Input : Batch of varying length sequences
Output : Tensor containing a hextuple <u1, u1_length, u2, u2_length, u3, u3_length>
'''
def batchify(batch):
    # input is a list of Triple objects
    bt_siz = len(batch)
    # sequence length only affects the memory requirement, otherwise longer is better
    pad_idx, max_seq_len = 10003, 160

    u1_batch, u2_batch, u3_batch = [], [], []
    u1_lens, u2_lens, u3_lens = np.zeros(bt_siz, dtype=int), np.zeros(bt_siz, dtype=int), np.zeros(bt_siz, dtype=int)

    # these store the max sequence lengths for the batch
    l_u1, l_u2, l_u3 = 0, 0, 0
    for i, (d, cl_u1, cl_u2, cl_u3) in enumerate(batch):
        cl_u1 = min(cl_u1, max_seq_len)
        cl_u2 = min(cl_u2, max_seq_len)
        cl_u3 = min(cl_u3, max_seq_len)

        if cl_u1 > l_u1:
            l_u1 = cl_u1
        u1_batch.append(torch.LongTensor(d.u1))
        u1_lens[i] = cl_u1

        if cl_u2 > l_u2:
            l_u2 = cl_u2
        u2_batch.append(torch.LongTensor(d.u2))
        u2_lens[i] = cl_u2

        if cl_u3 > l_u3:
            l_u3 = cl_u3
        u3_batch.append(torch.LongTensor(d.u3))
        u3_lens[i] = cl_u3

    t1, t2, t3 = u1_batch, u2_batch, u3_batch

    u1_batch = Variable(torch.ones(bt_siz, l_u1).long() * pad_idx)
    u2_batch = Variable(torch.ones(bt_siz, l_u2).long() * pad_idx)
    u3_batch = Variable(torch.ones(bt_siz, l_u3).long() * pad_idx)
    end_tok = torch.LongTensor([2])

    for i in range(bt_siz):
        seq1, cur1_l = t1[i], t1[i].size(0)
        if cur1_l <= l_u1:
            u1_batch[i, :cur1_l].data.copy_(seq1[:cur1_l])
        else:
            u1_batch[i, :].data.copy_(torch.cat((seq1[:l_u1-1], end_tok), 0))

        seq2, cur2_l = t2[i], t2[i].size(0)
        if cur2_l <= l_u2:
            u2_batch[i, :cur2_l].data.copy_(seq2[:cur2_l])
        else:
            u2_batch[i, :].data.copy_(torch.cat((seq2[:l_u2-1], end_tok), 0))

        seq3, cur3_l = t3[i], t3[i].size(0)
        if cur3_l <= l_u3:
            u3_batch[i, :cur3_l].data.copy_(seq3[:cur3_l])
        else:
            u3_batch[i, :].data.copy_(torch.cat((seq3[:l_u3-1], end_tok), 0))

    sort1, sort2, sort3 = np.argsort(u1_lens*-1), np.argsort(u2_lens*-1), np.argsort(u3_lens*-1)
    # cant call use_cuda here because this function block is used in threading calls

    return u1_batch[sort1, :], u1_lens[sort1], u2_batch[sort2, :], u2_lens[sort2], u3_batch[sort3, :], u3_lens[sort3]



'''
Function to convert tensor of sequnces of token ids to sentences
'''
def id_to_sentence(x, inv_dict, greedy=False):
    sents = []
    inv_dict[10003] = '<pad>'
    for li in x:
        if not greedy:
            scr = li[1]
            seq = li[0]
        else:
            scr = 0
            seq = li
        sent = []
        for i in seq:
            sent.append(inv_dict[i])
            if i == 2:
                break
        sents.append((" ".join(sent), scr))
    return sents


'''
Function to initialize all the parameters of the model.
According to the paper, rnn weights are orthogonally initialized, all other parameters are from a gaussian distribution with 0 mean and sd of 0.01.
'''
def init_param(model):
    for name, param in model.named_parameters():
        # skip over the embeddings so that the padding index ones are 0
        if 'embed' in name:
            continue
        elif ('rnn' in name or 'lm' in name) and len(param.size()) >= 2:
            init.orthogonal(param)
        else:
            init.normal(param, 0, 0.01)

'''
Function to calculate the liklihood of a given output sentence
'''
def get_sent_ll(u3, u3_lens, model, criteria, ses_encoding):
    preds, _ = model.dec([ses_encoding, u3, u3_lens])
    preds = preds[:, :-1, :].contiguous().view(-1, preds.size(2))
    u3 = u3[:, 1:].contiguous().view(-1)
    loss = criteria(preds, u3).data[0]
    target_tokens = u3.ne(10003).long().sum().data[0]
    return -1*loss/target_tokens


def calc_valid_loss(data_loader, criteria, model):
    model.eval()
    cur_tc = model.dec.get_teacher_forcing()
    model.dec.set_teacher_forcing(True)
    # we want to find the perplexity or likelihood of the provided sequence

    valid_loss, num_words = 0, 0
    for i_batch, sample_batch in enumerate(tqdm(data_loader)):
        preds, lmpreds = model(sample_batch)
        u3 = sample_batch[4]
        if use_cuda:
            u3 = u3.cuda()
        preds = preds[:, :-1, :].contiguous().view(-1, preds.size(2))
        u3 = u3[:, 1:].contiguous().view(-1)
        # do not include the lM loss, exp(loss) is perplexity
        loss = criteria(preds, u3)
        num_words += u3.ne(10003).long().sum().data[0]
        valid_loss += loss.data[0]

    model.train()
    model.dec.set_teacher_forcing(cur_tc)

    return valid_loss/num_words


def uniq_answer(fil):
    uniq = Counter()
    with open(fil + '_result.txt', 'r') as fp:
        all_lines=  fp.readlines()
        for line in all_lines:
            resp = line.split("    |    ")
            uniq[resp[1].strip()] += 1
    print('uniq', len(uniq), 'from', len(all_lines))
    print('---all---')
    for s in uniq.most_common():
        print(s)


def sort_key(temp, mmi):
    if mmi:
        lambda_param = 0.25
        return temp[1] - lambda_param*temp[2] + len(temp[0])*0.1
    else:
        return temp[1]/len(temp[0])**0.7

def max_out(x):
    # make sure s2 is even and that the input is 2 dimension
    if len(x.size()) == 2:
        s1, s2 = x.size()
        x = x.unsqueeze(1)
        x = x.view(s1, s2 // 2, 2)
        x, _ = torch.max(x, 2)

    elif len(x.size()) == 3:
        s1, s2, s3 = x.size()
        x = x.unsqueeze(1)
        x = x.view(s1, s2, s3 // 2, 2)
        x, _ = torch.max(x, 3)

    return x
