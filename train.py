import argparse
import time
import os
#from tqdm import tqdm

import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader

from model import *
from util import *
from dataset import *
from collections import Counter

use_cuda = torch.cuda.is_available()

#to keep reproducability in experiments
torch.manual_seed(123)
np.random.seed(123)
if use_cuda:
    torch.cuda.manual_seed(123)

'''
Function to load the model if bootstrapping is enabled
'''
def load_model_state(mdl, fl):
    saved_state = torch.load(fl)
    mdl.load_state_dict(saved_state)

'''
The training function
'''
def train(options, model):
    model.train()
    optimizer = optim.Adam(model.parameters(), options.lr)
    if options.btstrp:
        load_model_state(model, options.btstrp + "_mdl.pth")
        load_model_state(optimizer, options.btstrp + "_opti_st.pth")
    else:
        init_param(model)
    print("Model built and initialized!")
    if options.toy:
        train_dataset, valid_dataset = MovieTriples('train', 1000), MovieTriples('valid', 100)
    else:
        train_dataset, valid_dataset = MovieTriples('train'), MovieTriples('valid')

    train_dataloader = DataLoader(train_dataset, batch_size=options.bt_siz, shuffle=True, num_workers=2,
                                  collate_fn=batchify)
    valid_dataloader = DataLoader(valid_dataset, batch_size=options.bt_siz, shuffle=True, num_workers=2,
                                  collate_fn=batchify)

    print("Dataset loaded!")
    print("Training set {} Validation set {}".format(len(train_dataset), len(valid_dataset)))


    criteria = nn.CrossEntropyLoss(ignore_index=10003, size_average=False)
    if use_cuda:
        criteria.cuda()

    best_vl_loss, patience, batch_id = 10000, 0, 0
    print("Training started!")
    for i in range(options.epoch):
        if patience == options.patience:
            break
        tr_loss, tlm_loss, num_words = 0, 0, 0
        strt = time.time()

        for i_batch, sample_batch in enumerate(tqdm(train_dataloader)):
            if not options.teacher:
                new_tc_ratio = 2100.0/(2100.0 + math.exp(i_batch/2100.0))
                model.dec.set_tc_ratio(new_tc_ratio)

            preds, lmpreds = model(sample_batch)
            u3 = sample_batch[4]
            if use_cuda:
                u3 = u3.cuda()

            preds = preds[:, :-1, :].contiguous().view(-1, preds.size(2))
            u3 = u3[:, 1:].contiguous().view(-1)

            loss = criteria(preds, u3)
            target_tokens = u3.ne(10003).long().sum().data[0]

            num_words += target_tokens
            tr_loss += loss.data[0]
            loss = loss/target_tokens

            if options.lm:
                lmpreds = lmpreds[:, :-1, :].contiguous().view(-1, lmpreds.size(2))
                lm_loss = criteria(lmpreds, u3)
                tlm_loss += lm_loss.data[0]
                lm_loss = lm_loss/target_tokens

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            if options.lm:
                lm_loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 1)
            optimizer.step()

            batch_id += 1

        vl_loss = calc_valid_loss(valid_dataloader, criteria, model)
        print("Training loss {} lm loss {} Valid loss {}".format(tr_loss/num_words, tlm_loss/num_words, vl_loss))
        print("epoch {} took {} mins".format(i+1, (time.time() - strt)/60.0))
        print("tc ratio", model.dec.get_tc_ratio())
        if vl_loss < best_vl_loss or options.toy:
            #f_mod = open(os.path.abspath(options.model_path + '/' + options.name + '_model.pth'), 'w+')
            #f_opt = open(os.path.abspath(options.model_path + '/' + options.name + '_optimer_state.pth'), 'w+')
            torch.save(model.state_dict(), options.model_path + '/' + options.name + '_model.pth')
            torch.save(optimizer.state_dict(), options.model_path + '/' + options.name + '_optimer_state.pth')
            best_vl_loss = vl_loss
            patience = 0
        else:
            patience += 1




def generate(model, ses_encoding, options):
    diversity_rate = 2
    antilm_param = 10
    beam = options.beam

    n_candidates, final_candids = [], []
    candidates = [([1], 0, 0)]
    gen_len, max_gen_len = 1, 20

    # we provide the top k options/target defined each time
    while gen_len <= max_gen_len:
        for c in candidates:
            seq, pts_score, pt_score = c[0], c[1], c[2]
            _target = Variable(torch.LongTensor([seq]), volatile=True)
            dec_o, dec_lm = model.dec([ses_encoding, _target, [len(seq)]])
            dec_o = dec_o[:, :, :-1]

            op = F.log_softmax(dec_o, 2, 5)
            op = op[:, -1, :]
            topval, topind = op.topk(beam, 1)

            if options.lm:
                dec_lm = dec_lm[:, :, :-1]
                lm_op = F.log_softmax(dec_lm, 2, 5)
                lm_op = lm_op[:, -1, :]

            for i in range(beam):
                ctok, cval = topind.data[0, i], topval.data[0, i]
                if options.lm:
                    uval = lm_op.data[0, ctok]
                    if dec_lm.size(1) > antilm_param:
                        uval = 0.0
                else:
                    uval = 0.0

                if ctok == 2:
                    list_to_append = final_candids
                else:
                    list_to_append = n_candidates

                list_to_append.append((seq + [ctok], pts_score + cval - diversity_rate*(i+1), pt_score + uval))

        n_candidates.sort(key=lambda temp: sort_key(temp, options.mmi), reverse=True)
        candidates = copy.copy(n_candidates[:beam])
        n_candidates[:] = []
        gen_len += 1

    final_candids = final_candids + candidates
    final_candids.sort(key=lambda temp: sort_key(temp, options.mmi), reverse=True)

    return final_candids[:beam]





# sample a sentence from the test set by using beam search
def inference_beam(dataloader, model, inv_dict, options):
    criteria = nn.CrossEntropyLoss(ignore_index=10003, size_average=False)
    if use_cuda:
        criteria.cuda()

    cur_tc = model.dec.get_teacher_forcing()
    model.dec.set_teacher_forcing(True)
    fout1 = open(options.res_path + '/' + options.name + "_groundtruth.txt",'w')
    fout2 = open(options.res_path + '/' + options.name + "_model_preds.txt",'w')
    load_model_state(model, options.name + "_mdl.pth")
    model.eval()

    test_ppl = calc_valid_loss(dataloader, criteria, model)
    print("test preplexity is:{}".format(test_ppl))

    for i_batch, sample_batch in enumerate(dataloader):
        u1, u1_lens, u2, u2_lens, u3, u3_lens = sample_batch[0], sample_batch[1], sample_batch[2], sample_batch[3], \
                                                sample_batch[4], sample_batch[5]

        if use_cuda:
            u1 = u1.cuda()
            u2 = u2.cuda()
            u3 = u3.cuda()

        o1, o2 = model.utt_enc((u1, u1_lens)), model.utt_enc((u2, u2_lens))
        #qu_seq = torch.cat((o1, o2), 2)
        # if we need to decode the intermediate queries we may need the hidden states
        if options.seq2seq:
            qu_seq = torch.cat((o1, o2), 2)
            final_session_o = qu_seq
        else:
            qu_seq = torch.cat((o1, o2), 1)
            final_session_o = model.intutt_enc(qu_seq)
        #final_session_o = model.intutt_enc(qu_seq)
        # forward(self, ses_encoding, x=None, x_lens=None, beam=5 ):
        #sent = generate(model, final_session_o, options)
        #pt = id_to_sentence(sent, inv_dict)
        # greedy true for below because only beam generates a tuple of sequence and probability
        #gt = id_to_sentence(u3.data.cpu().numpy(), inv_dict, True)
        #print(pt, gt)
        for k in range(options.bt_siz):
            sent = generate(model, final_session_o[k, :, :].unsqueeze(0), options)
            pt = id_to_sentence(sent, inv_dict)
            # greedy true for below because only beam generates a tuple of sequence and probability
            gt = id_to_sentence(u3[k, :].unsqueeze(0).data.cpu().numpy(), inv_dict, True)
            fout1.write(str(gt[0]) + "\n")
            fout2.write(str(pt[0][0]) + "\n")
            fout1.flush()
            fout2.flush()
            '''
            if not options.pretty:

                print("Ground truth {} {} \n".format(gt, get_sent_ll(u3[k, :].unsqueeze(0), u3_lens[k:k+1], model, criteria, final_session_o[k, :, :].unsqueeze(0))))
            else:
                print(gt[0], "|", pt[0][0])
                '''
    model.dec.set_teacher_forcing(cur_tc)
    fout.close()





def main():
    print('torch version {}'.format(torch.__version__))
    _dict_file = './data/word_summary.pkl'
    # we use a common dict for all test, train and validation

    with open(_dict_file, 'rb') as fp2:
        dict_data = pickle.load(fp2)
    # dictionary data is like ('</s>', 2, 588827, 785135)
    # so i believe that the first is the ids are assigned by frequency
    # thinking to use a counter collection out here maybe
    inv_dict = {}
    for x in dict_data:
        tok, f, _, _ = x
        inv_dict[f] = tok

    parser = argparse.ArgumentParser(description='HRED parameter options')
    parser.add_argument('-n', dest='name', help='enter suffix for model files')
    parser.add_argument('-res_path', dest='res_path', default='./results', help='enter the path in which you want to store the results')
    parser.add_argument('-model_path', dest='model_path', default='./models', help='enter the path in which you want to store the model state')
    parser.add_argument('-e', dest='epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('-pt', dest='patience', type=int, default=-1, help='validtion patience for early stopping default none')
    parser.add_argument('-tf', dest='teacher', action='store_true', default=False, help='default teacher forcing')
    parser.add_argument('-bi', dest='bidi', action='store_true', default=False, help='bidirectional enc/decs')
    parser.add_argument('-test', dest='test', action='store_true', default=False, help='only test or inference')
    parser.add_argument('-shrd_dec_emb', dest='shrd_dec_emb', action='store_true', default=False, help='shared embedding in/out for decoder')
    parser.add_argument('-btstrp', dest='btstrp', default=None, help='bootstrap/load parameters give name')
    parser.add_argument('-lm', dest='lm', action='store_true', default=False, help='enable a RNN language model joint training as well')
    parser.add_argument('-toy', dest='toy', action='store_true', default=False, help='loads only 1000 training and 100 valid for testing')
    parser.add_argument('-pretty', dest='pretty', action='store_true', default=False, help='pretty print inference')
    parser.add_argument('-mmi', dest='mmi', action='store_true', default=False, help='Using the mmi anti-lm for ranking beam')
    parser.add_argument('-s2s', dest='seq2seq', action='store_true', default=False, help='Using baseline seq2seq model')
    parser.add_argument('-drp', dest='drp', type=float, default=0.3, help='dropout probability used all throughout')
    parser.add_argument('-nl', dest='num_lyr', type=int, default=1, help='number of enc/dec layers(same for both)')
    parser.add_argument('-lr', dest='lr', type=float, default=0.001, help='learning rate for optimizer')
    parser.add_argument('-bs', dest='bt_siz', type=int, default=100, help='batch size')
    parser.add_argument('-bms', dest='beam', type=int, default=1, help='beam size for decoding')
    parser.add_argument('-vsz', dest='vocab_size', type=int, default=10004, help='size of vocabulary')
    parser.add_argument('-esz', dest='emb_size', type=int, default=300, help='embedding size enc/dec same')
    parser.add_argument('-uthid', dest='ut_hid_size', type=int, default=600, help='encoder utterance hidden state')
    parser.add_argument('-seshid', dest='ses_hid_size', type=int, default=1200, help='encoder session hidden state')
    parser.add_argument('-dechid', dest='dec_hid_size', type=int, default=600, help='decoder hidden state')
    parser.add_argument('-embed', dest='use_embed', action='store_true', default=False, help='use pretrained word embeddings for the encoder')

    options = parser.parse_args()
    print(options)
    if not os.path.exists(options.res_path):
        os.makedirs(options.res_path)
    if not os.path.exists(options.model_path):
        os.makedirs(options.model_path)

    model = HSeq2seq(options)
    if use_cuda:
        model.cuda()

    if not options.test:
        train(options, model)
    else:
        if options.toy:
            test_dataset = MovieTriples('test', 100)
        else:
            test_dataset = MovieTriples('test')


        test_dataloader = DataLoader(test_dataset, options.bt_siz, shuffle=True, num_workers=2, collate_fn=batchify)
        inference_beam(test_dataloader, model, inv_dict, options)
        uniq_answer(options.name)

main()
