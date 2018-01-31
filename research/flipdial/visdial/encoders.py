import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import archies

PAD_ID = 0
EOS_ID = 1
UNK_ID = 3

        
class block_encoder(nn.Module):
    """ Encodes a block of dialogue """
    def __init__(self, input_tag, condition_tag, nhid, emsize, seqlen, imgfeatsize, fbase, w2v_emsize, w2vcap):
        super(block_encoder, self).__init__()

        self.itag = input_tag
        self.ctag = condition_tag
        
        self.sizes = {}
        self.sizes['latent'] = nhid
        self.sizes['emb'] = emsize
        self.sizes['seqlen'] = seqlen
        self.sizes['fbase'] = fbase
        self.sizes['imgfeat'] = imgfeatsize

        assert (condition_tag == 'img' or condition_tag == 'img_cap' or condition_tag == 'cap' or condition_tag == 'none')
        assert sum([int(i in input_tag) for i in condition_tag.split('_')])==0, 'input and condition tags cannot overlap'
        
        self.architags = {}
        self.tag2mod = {}
        self.enc = nn.ModuleList()
        m=0
        
        tag = 'dialogblock_(20,' + str(seqlen) + ',' + str(emsize) + ')' 
        self.architags['dialogblock'] = tag
        self.enc.append(archies.enc_archies[tag])
        self.tag2mod['dialogblock'] = m
        m+=1
        if not self.ctag == 'none':
            self.enc.append(nn.Sequential(nn.Conv2d(fbase*8, fbase*4, 1, 1, 0, bias=False), nn.BatchNorm2d(fbase*4), nn.ReLU(True)))
            self.tag2mod['dialogblock_fc'] = m
            m+=1
        
        if 'cap' in self.ctag:
            if w2vcap:
                tag = 'item_(1,' + str(seqlen) + ',' + str(w2v_emsize) + ')' 
            else:
                tag = 'item_(1,' + str(seqlen) + ',' + str(emsize) + ')' 
            self.architags['cap'] = tag
            self.enc.append(archies.enc_archies[tag])
            self.tag2mod['cap'] = m
            m+=1
        if 'img' in self.ctag:
            self.enc.append(nn.Sequential(nn.Conv2d(fbase*8, fbase*4, 1, 1, 0, bias=False),nn.BatchNorm2d(fbase*4),nn.ReLU(True)))
            self.tag2mod['img_fc'] = m
            m+=1

        self.architags['joint_p'] = 'joint_(' + str(fbase*4) + ',' + str(8) + ',' + str(8) + ')'
        self.enc.append(archies.enc_archies[self.architags['joint_p']])
        self.tag2mod['joint_p'] = m
        m+=1
        self.architags['joint_q'] = 'joint_(' + str(fbase*4) + ',' + str(8) + ',' + str(8) + ')'
        self.enc.append(archies.enc_archies[self.architags['joint_q']])
        self.tag2mod['joint_q'] = m
        m+=1
       
        if not self.ctag == 'none':
            self.c1_p = nn.Sequential(nn.Conv2d(fbase * 16, self.sizes['latent'], 4, 1, 0, bias=False), nn.BatchNorm2d(self.sizes['latent']), nn.ReLU(True))
            self.c2_p = nn.Sequential(nn.Conv2d(fbase * 16, self.sizes['latent'], 4, 1, 0, bias=False), nn.BatchNorm2d(self.sizes['latent']), nn.ReLU(True)) 
        # c1, c2 size: latentdim x 1 x 1
        
        self.c1_q = nn.Sequential(nn.Conv2d(fbase * 16, self.sizes['latent'], 4, 1, 0, bias=False), nn.BatchNorm2d(self.sizes['latent']), nn.ReLU(True))
        self.c2_q = nn.Sequential(nn.Conv2d(fbase * 16, self.sizes['latent'], 4, 1, 0, bias=False), nn.BatchNorm2d(self.sizes['latent']), nn.ReLU(True))

    def forward(self, x_dict={}, evaluation=False):
        output = {}

        enc_condition=None
        if 'cap' in self.ctag:
            enc_condition = self.enc[self.tag2mod['cap']](x_dict['cap'])
        
        if 'img' in self.ctag:
            if isinstance(enc_condition, Variable):
                enc_condition = self.enc[self.tag2mod['img_fc']](torch.cat((enc_condition, x_dict['img'].view_as(enc_condition)), 1))
            else:
                enc_condition = x_dict['img'].view(x_dict['img'].size(0), self.sizes['fbase']*4, 8,8)

        if not isinstance(enc_condition, Variable):
            enc_q = self.enc[self.tag2mod['dialogblock']](x_dict['dialogblock'])
            output['mu_p'] = Variable(torch.zeros(x_dict['dialogblock'].size(0), self.sizes['latent'], 1, 1).type_as(enc_q.data), volatile=evaluation)
            output['logvar_p'] = Variable(torch.zeros(x_dict['dialogblock'].size(0), self.sizes['latent'], 1, 1).type_as(enc_q.data), volatile=evaluation)
        else:
            enc_p = self.enc[self.tag2mod['joint_p']](enc_condition)
            output['mu_p'] = self.c1_p(enc_p)
            output['logvar_p'] = self.c2_p(enc_p)
            enc_q = self.enc[self.tag2mod['dialogblock_fc']](torch.cat((enc_condition, self.enc[self.tag2mod['dialogblock']](x_dict['dialogblock'])), 1))
        
        #if not evaluation        
        enc_q = self.enc[self.tag2mod['joint_q']](enc_q)
        
        output['mu_q'] = self.c1_q(enc_q)
        output['logvar_q'] = self.c2_q(enc_q)
        
        return enc_condition, output

class item_encoder(nn.Module):
    """ Encodes a sentence of dialogue """
    def __init__(self, input_tag, condition_tag, nhid, emsize, seqlen, imgfeatsize, fbase, w2v_emsize, w2vcap):
        super(item_encoder, self).__init__()

        self.itag = input_tag
        self.ctag = condition_tag
        
        self.sizes = {}
        self.sizes['latent'] = nhid
        self.sizes['emb'] = emsize
        self.sizes['seqlen'] = seqlen
        self.sizes['fbase'] = fbase
        self.sizes['imgfeat'] = imgfeatsize
        self.sizes['w2v_emb'] = w2v_emsize
        
        assert sum([int(i in input_tag) for i in condition_tag.split('_')])==0, 'input and condition tags cannot overlap'
        
        self.architags = {}
        self.tag2mod = {}
        self.enc = nn.ModuleList()
        m=0
        if 'answer' in self.itag or 'answer' in self.ctag:
            tag = 'item_(1,' + str(seqlen) + ',' + str(emsize) + ')' 
            self.architags['answer'] = tag
            self.enc.append(archies.enc_archies[tag])
            self.tag2mod['answer'] = m
            m+=1
        if 'question' in self.itag or 'question' in self.ctag:
            tag = 'item_(1,' + str(seqlen) + ',' + str(emsize) + ')' 
            self.architags['question'] = tag
            self.enc.append(archies.enc_archies[tag])
            self.tag2mod['question'] = m
            m+=1
        #if sum([int(i=='question' or i=='answer') for i in (input_tag + '_' + condition_tag).split('_')]) == 2:
        if 'question' in input_tag and 'answer' in input_tag:
            self.enc.append(nn.Sequential(nn.Conv2d(fbase*8, fbase*4, 1, 1, 0, bias=False), nn.BatchNorm2d(fbase*4), nn.ReLU(True)))
            self.tag2mod['question_fc'] = m
            m+=1
        
        if 'img' in self.ctag:
            self.enc.append(nn.Sequential(nn.Conv2d(fbase*8, fbase*4, 1, 1, 0, bias=False), nn.BatchNorm2d(fbase*4), nn.ReLU(True)))
            self.tag2mod['img_fc'] = m
            m+=1
        if 'cap' in self.ctag:
            if w2vcap:
                tag = 'item_(1,' + str(seqlen) + ',' + str(w2v_emsize) + ')' 
            else:
                tag = 'item_(1,' + str(seqlen) + ',' + str(emsize) + ')' 
            self.architags['cap'] = tag
            self.enc.append(archies.enc_archies[tag])
            self.tag2mod['cap'] = m
            m+=1
            self.enc.append(nn.Sequential(nn.Conv2d(fbase*8, fbase*4, 1, 1, 0, bias=False), nn.BatchNorm2d(fbase*4), nn.ReLU(True)))
            self.tag2mod['cap_fc'] = m
            m+=1
        if 'history' in self.ctag:
            tag = 'history_(20,' + str(seqlen) + ',' + str(emsize) + ')' 
            self.architags['history'] = tag
            self.enc.append(archies.enc_archies[tag])
            self.tag2mod['history'] = m
            m+=1
            self.enc.append(nn.Sequential(nn.Conv2d(fbase*8, fbase*4, 1, 1, 0, bias=False), nn.BatchNorm2d(fbase*4), nn.ReLU(True)))
            self.tag2mod['history_fc'] = m
            m+=1

        self.enc.append(nn.Sequential(nn.Conv2d(fbase*8, fbase*4, 1, 1, 0, bias=False), nn.BatchNorm2d(fbase*4), nn.ReLU(True)))
        self.tag2mod['joint_pq'] = m
        m+=1

        self.architags['joint_p'] = 'joint_(' + str(fbase*4) + ',' + str(8) + ',' + str(8) + ')'
        self.enc.append(archies.enc_archies[self.architags['joint_p']])
        self.tag2mod['joint_p'] = m
        m+=1
        self.architags['joint_q'] = 'joint_(' + str(fbase*4) + ',' + str(8) + ',' + str(8) + ')'
        self.enc.append(archies.enc_archies[self.architags['joint_q']])
        self.tag2mod['joint_q'] = m
        m+=1
        
        self.c1_p = nn.Sequential(nn.Conv2d(fbase * 16, self.sizes['latent'], 4, 1, 0, bias=False), nn.BatchNorm2d(self.sizes['latent']), nn.ReLU(True))
        self.c2_p = nn.Sequential(nn.Conv2d(fbase * 16, self.sizes['latent'], 4, 1, 0, bias=False), nn.BatchNorm2d(self.sizes['latent']), nn.ReLU(True))

        # c1, c2 size: latentdim x 1 x 1
        
        self.c1_q = nn.Sequential(nn.Conv2d(fbase * 16, self.sizes['latent'], 4, 1, 0, bias=False), nn.BatchNorm2d(self.sizes['latent']), nn.ReLU(True))
        self.c2_q = nn.Sequential(nn.Conv2d(fbase * 16, self.sizes['latent'], 4, 1, 0, bias=False), nn.BatchNorm2d(self.sizes['latent']), nn.ReLU(True))


    def forward(self, x_dict={}, evaluation=False):
        output = {}

        enc_condition = None

        if 'question' in self.ctag:
            enc_condition = self.enc[self.tag2mod['question']](x_dict['question'])
        elif 'answer' in self.ctag:
            enc_condition = self.enc[self.tag2mod['answer']](x_dict['answer'])

        if 'cap' in self.ctag:
            if isinstance(enc_condition, Variable):
                enc_condition = F.relu(self.enc[self.tag2mod['cap_fc']](torch.cat((enc_condition, self.enc[self.tag2mod['cap']](x_dict['cap'])), 1)))
            else:
                enc_condition = self.enc[self.tag2mod['cap']](x_dict['cap'])
        
        if 'img' in self.ctag:
            if isinstance(enc_condition, Variable):
                enc_condition = F.relu(self.enc[self.tag2mod['img_fc']](torch.cat((enc_condition, x_dict['img'].view_as(enc_condition)),1)))
            else:
                enc_condition = x_dict['img'].view(x_dict['img'].size(0), self.sizes['fbase']*4, 8,8)

        if 'history' in self.ctag:
            if isinstance(enc_condition, Variable):
                enc_condition = F.relu(self.enc[self.tag2mod['history_fc']](torch.cat((enc_condition, self.enc[self.tag2mod['history']](x_dict['history'])), 1)))
            else:
                enc_condition = self.enc[self.tag2mod['history']](x_dict['history'])

        enc_p = self.enc[self.tag2mod['joint_p']](enc_condition)
        output['mu_p'] = self.c1_p(enc_p)
        output['logvar_p'] = self.c2_p(enc_p)
        
        if not evaluation:
            if 'answer' in self.itag and 'question' in self.itag:
                enc_q = F.relu(self.enc[self.tag2mod['question_fc']](torch.cat((self.enc[self.tag2mod['question']](x_dict['question']), self.enc[self.tag2mod['answer']](x_dict['answer'])), 1)))
            elif 'answer' in self.itag:
                enc_q = self.enc[self.tag2mod['answer']](x_dict['answer'])
            elif 'question' in self.itag:
                enc_q = self.enc[self.tag2mod['question']](x_dict['question'])
            
            enc_q = F.relu(self.enc[self.tag2mod['joint_pq']](torch.cat((enc_condition, enc_q), 1)))
            enc_q = self.enc[self.tag2mod['joint_q']](enc_q)
        
            output['mu_q'] = self.c1_q(enc_q)
            output['logvar_q'] = self.c2_q(enc_q)
        
        return enc_condition, output 
