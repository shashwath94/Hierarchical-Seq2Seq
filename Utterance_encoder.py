import torch.nn as nn, torch, copy, tqdm, math
from torch.autograd import Variable
import torch.nn.functional as F
import pickle
import numpy as np
use_cuda = torch.cuda.is_available()






# encode each sentence utterance into a single vector
class UtteranceEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size, options):
        super(UtteranceEncoder, self).__init__()
        self.use_embed = options.use_embed
        self.hid_size = hid_size
        self.num_lyr = options.num_lyr
        self.drop = nn.Dropout(options.drp)
        self.direction = 2 if options.bidi else 1
        # by default they requires grad is true
        self.embed = nn.Embedding(vocab_size, emb_size, padding_idx=10003, sparse=False)
        if self.use_embed:
            pretrained_weight = self.load_embeddings(vocab_size, emb_size)
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.rnn = nn.GRU(input_size=emb_size, hidden_size=hid_size,
                          num_layers=self.num_lyr, bidirectional=options.bidi, batch_first=True, dropout=options.drp)

    def forward(self, inp):
        x, x_lengths = inp[0], inp[1]
        bt_siz, seq_len = x.size(0), x.size(1)
        h_0 = Variable(torch.zeros(self.direction * self.num_lyr, bt_siz, self.hid_size))
        if use_cuda:
            h_0 = h_0.cuda()
        token_emb = self.embed(x)
        token_emb = self.drop(token_emb)
        token_emb = torch.nn.utils.rnn.pack_padded_sequence(token_emb, x_lengths, batch_first=True)
        gru_out, gru_hid = self.rnn(token_emb, h_0)
        # assuming dimension 0, 1 is for layer 1 and 2, 3 for layer 2
        if self.direction == 2:
            gru_hids = []
            for i in range(self.num_lyr):
                x_hid_temp, _ = torch.max(gru_hid[2*i:2*i + 2, :, :], 0, keepdim=True)
                gru_hids.append(x_hid_temp)
            gru_hid = torch.cat(gru_hids, 0)
        # gru_out, _ = torch.nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        # using gru_out and returning gru_out[:, -1, :].unsqueeze(1) is wrong coz its all 0s careful! it doesn't adjust for variable timesteps

        gru_hid = gru_hid[self.num_lyr-1, :, :].unsqueeze(0)
        # take the last layer of the encoder GRU
        gru_hid = gru_hid.transpose(0, 1)

        return gru_hid

    def load_embeddings(self, vocab_size, emb_size):
        vocab_file = './data/word_summary.pkl'
        embed_file = './data/embeddings/glove.840B.300d.txt'
        vocab = {}
        embeddings_index = {}
        with open(vocab_file, 'rb') as fp2:
            dict_data = pickle.load(fp2)

        for x in dict_data:
            tok, f, _, _ = x
            vocab[tok] = f

        #f = open(embed_file)
        with open(embed_file, 'r') as fp:
            f = fp.readlines()
        for line in f:
            values = line.split()
            word = values[0]
            if len(values) > 301:
                continue
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

        embedding_matrix = np.zeros((vocab_size, emb_size))
        for word, i in vocab.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
		#print(embedding_vector)
                embedding_matrix[i] = embedding_vector
        print(embedding_matrix.shape)
        return embedding_matrix
