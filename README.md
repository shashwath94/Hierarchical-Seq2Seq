# Hierarchical-Seq2Seq
A PyTorch implementation of the hierarchical encoder-decoder architecture (HRED) introduced in Sordoni et al (2015). It is a hierarchical encoder-decoder architecture for modeling conversation triples. This version of the model is built for the MovieTriples dataset.

Link to related papers - <a href='https://arxiv.org/abs/1507.02221'> HRED</a>, <a href='https://arxiv.org/pdf/1507.04808.pdf'> Modified HRED</a>

<h4> Dataset details </h4>
This model is trained for the MovieTriples dataset. This dataset is available on request to the lead author of the second paper mentioned above and is not publicly available. 

Training commands 

For baseline seq2seq model - ` python3 train.py -n seq2seq -tf -bms 20 -bs 100 -e 80 -seshid 300 -uthid 300 -drp 0.4 -lr 0.0005 -s2s -pt 3`

For HRED model - `python3 train.py -n HRED -tf -bms 20 -bs 100 -e 80 -seshid 300 -uthid 300 -drp 0.4 -lr 0.0005 -pt 3`

For Bi-HRED + Language model objective (inverse sigmoid teacher forcing rate decay) - `python3 train.py -n BiHRED+LM -bi -lm -nl 2 -lr 0.0003 -e 80 -seshid 300 -uthid 300  -bs 10 -pt 3`

For Bi-HRED + Language model objective (Ful teacher forcing) - python3 train.py -n model3 -nl 2 -bi -lm -drp 0.4 -e 25 -seshid 300 -uthid 300 -lr 0.0001 -bs 100 -tf

At test time, we use beam search decoding with beam size set at 20. For reranking the candidates during beam search, we use the MMI Anit-LM following the method in <a href='https://arxiv.org/pdf/1510.03055.pdf'> paper </a>

Test command - Just add the following flags for testing the model. `-test -mmi -bms 50`

To perform a sanity check on the model, train the model on a small subset of the dataset with the flag `-toy`. It should overfit with a training error of 0.5.
