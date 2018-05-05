'''
This python module contains classes to construct a dataset from the raw dataset(train, valid, test) data files
'''

import torch
import copy
import pickle
from torch.utils.data import Dataset
from torch.autograd import Variable
import numpy as np

#check to see if GPU is available
use_cuda = torch.cuda.is_available()




'''
Wrapper class for creating a single training example from every line in the raw dataset.
This class iterates over row of the raw dataset and creates a triple of <u1, u2, u3>.

'''
class Triple:
    def __init__(self, item):
        self.u1, self.u2, self.u3 = [], [], []
        cur_list, i = [], 0
        for d in item:
            cur_list.append(d)
            if d == 2:
                if i == 0:
                    self.u1 = copy.copy(cur_list)
                    cur_list[:] = []
                elif i == 1:
                    self.u2 = copy.copy(cur_list)
                    cur_list[:] = []
                else:
                    self.u3 = copy.copy(cur_list)
                    cur_list[:] = []
                i += 1

    def __len__(self):
        return len(self.u1) + len(self.u2) + len(self.u3)

    def __repr__(self):
        return str(self.u1 + self.u2 + self.u3)

'''
Main dataset class
Each data sample is a triple <u1, u2, u3> where u1, u2, u3 are utterances in a conversation from a movie.
'''
class MovieTriples(Dataset):
    def __init__(self, data_type, length=None):
        if data_type == 'train':
            _file = './data/train_raw.pkl'
        elif data_type == 'valid':
            _file = './data/validate_raw.pkl'
        elif data_type == 'test':
            _file = './data/test_raw.pkl'
        self.utterance_data = []

        with open(_file, 'rb') as fp:
            data = pickle.load(fp)
            for d in data:
                self.utterance_data.append(Triple(d))
        # it helps in optimization that the batch be diverse, definitely helps!
        # self.utterance_data.sort(key=cmp_to_key(cmp_dialog))
        if length:
            self.utterance_data = self.utterance_data[2000:2000 + length]

    def __len__(self):
        return len(self.utterance_data)

    def __getitem__(self, idx):
        dialog = self.utterance_data[idx]
        return dialog, len(dialog.u1), len(dialog.u2), len(dialog.u3)
