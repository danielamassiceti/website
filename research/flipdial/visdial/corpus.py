#from PIL import Image
import torch
#from nltk import word_tokenize
#import inflect, re, os, json, gensim

#nc = inflect.engine()

# user-defined packages
#from utils import *

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.freq = {}
        self.PAD_ID = 0
        self.EOS_ID = 1
        self.GO_ID = 2
        self.UNK_ID = 3
	self.add_word('PAD'.encode('utf-8', 'ignore'))
	self.add_word('EOS'.encode('utf-8', 'ignore'))
        self.add_word('GO'.encode('utf-8', 'ignore'))
        self.add_word('UNK'.encode('utf-8', 'ignore'))  
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.freq[word] = 1
        else:
            self.freq[word] = self.freq[word] + 1
        
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.idx2word)

    def loadwordmodel(self, destfile):
        self.word_embs = torch.load(destfile)

    def filterbywordfreq(self):
        new_word2idx = {}
        new_idx2word = []
        new_freq = {}
        for idx in xrange(len(self.word2idx)):
            word = self.idx2word[idx]
            if self.freq[word] >=5 or idx in range(0,4):
                new_idx2word.append(word)
                new_word2idx[word] = len(new_idx2word)-1
                new_freq[word] = self.freq[word]
        self.idx2word = new_idx2word
        self.word2idx = new_word2idx
        self.freq = new_freq

def load_dictionary(filename):
    import sys
    sys.path.append('corpus.py')
    return torch.load(filename)

