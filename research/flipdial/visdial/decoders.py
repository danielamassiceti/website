import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import archies
from numpy import linalg as la


class block_decoder(nn.Module):
    """ Generate a block of dialogue given a sample from the latent space. """
    def __init__(self, dtag, nhid, emsize, seqlen, fbase, vocabsize):
        super(block_decoder, self).__init__()

        self.dtag = dtag

        self.sizes = {}
        self.sizes['latent'] = nhid
        self.sizes['emb'] = emsize
        self.sizes['seqlen'] = seqlen
        self.sizes['fbase'] = fbase
        self.sizes['vocab'] = vocabsize #targetsize
        
        self.architags = {}
        self.architags['dec1'] = 'latent_(' + str(self.sizes['latent']) + ',1,1)'
        self.dec1 = archies.dec_archies[self.architags['dec1']]
        self.dec_fc = nn.Sequential(nn.Conv2d(fbase*8, fbase*4, 1, 1, 0, bias=False), nn.BatchNorm2d(fbase*4), nn.ReLU(True))
        self.architags['dec2'] = 'latent_and_condition_(' + str(self.sizes['seqlen']) + ',8,8)_dialogblock'
        self.dec2 = archies.dec_archies[self.architags['dec2']]
        
        if 'autoreg' in dtag:
            self.architags['dec3'] = dtag + '_' + str(self.sizes['emb']) + '_' + str(self.sizes['emb']) + '_' + str(self.sizes['vocab'])
            self.dec3 = archies.dec_archies[self.architags['dec3']]
        
        self.tovocab = nn.Linear(self.sizes['emb'], self.sizes['vocab'], bias=False) 

    def forward(self, z, enc_condition, gt=None, fix='none'):
        # latentdim x 1 x 1
        if isinstance(enc_condition, Variable):
            dec_gt = self.dec2(self.dec_fc(torch.cat((enc_condition,self.dec1(z)), dim=1)))
        else:
            dec_gt = self.dec2(self.dec1(z))
        # 20 x seqlen x emsize

        bsz = dec_gt.size(0)
        nexchanges = dec_gt.size(1)
        seqlen = dec_gt.size(2)

        if 'autoreg' in self.dtag:
            dec_gt = dec_gt.permute(0,3,1,2).contiguous()
            if fix == 'hist_predQA' or fix == 'hist_predA':
                output = self.fillin_AR(dec_gt, gt, fix)
            elif fix == 'slotquestions':
                q_inds = torch.arange(0,nexchanges,2).type_as(gt.data).long()
                for m in self.dec3.children():
                    if isinstance(m, nn.Conv2d):
                        dec_gt.data[:,:,q_inds] = gt.select_index(1,q_inds).unsqueeze(1).permute(0,3,1,2).data
                    dec_gt = m(dec_gt)
                dec_gt.data[:,:,q_inds] = gt.select_index(1,q_inds).unsqueeze(1).permute(0,3,1,2).data
                output = self.tovocab(dec_gt.permute(0,2,3,1))
            elif fix == 'none':
                dec_gt = self.dec3(dec_gt)
                output = self.tovocab(dec_gt.permute(0,2,3,1))
        elif 'standard' in self.dtag:
            if fix == 'hist_predQA' or fix == 'hist_predA':
                output = self.fillin(dec_gt, gt, fix)
            elif fix == 'none':
                output = self.tovocab(dec_gt)
        
        return output
        #return self.output['tovocab'](dec_gt.view(-1, self.sizes['emb'])).view(z.size(0), -1, self.sizes['vocab'])
    
    def fillin(self, dec_gt, gt, fix):
        output = []
        if 'predQA' in fix:
            qa_inds = torch.arange(0,gt.size(1)).type_as(gt.data).long()
            for ind in qa_inds:
                if not ind == 0:
                    dec_gt.data[:,ind-1] = gt.data[:,ind-1]
                output.append(self.tovocab(dec_gt)[:,ind]) 
        elif 'predA' in fix:
            a_inds = torch.arange(1,gt.size(1),2).type_as(gt.data).long()
            for ind in a_inds:
                if ind==1:
                    dec_gt.data[:,ind-1] = gt.data[:,ind-1]
                else:
                    dec_gt.data[:,ind-2:ind] = gt.data[:,ind-2:ind]
                output.append(self.tovocab(dec_gt)[:,ind-1:ind+1]) 
        output = torch.stack(output,1)
        return output.view(output.size(0), -1, output.size(3), output.size(4))
    
class item_decoder(nn.Module):
    """ Generate a sentence of dialogue given a sample from the latent space. """
    def __init__(self, dtag, nhid, emsize, seqlen, fbase, vocabsize):
        super(item_decoder, self).__init__()

        self.dtag = dtag

        self.sizes = {}
        self.sizes['latent'] = nhid
        self.sizes['emb'] = emsize
        self.sizes['seqlen'] = seqlen
        self.sizes['fbase'] = fbase
        self.sizes['vocab'] = vocabsize #targetsize
        
        self.architags = {}
        self.architags['dec1'] = 'latent_(' + str(self.sizes['latent']) + ',1,1)'
        self.dec1 = archies.dec_archies[self.architags['dec1']]
        self.dec_fc = nn.Conv2d(fbase*8, fbase*4, 1, 1, 0, bias=False)
        self.architags['dec2'] = 'latent_and_condition_(' + str(self.sizes['seqlen']) + ',8,8)'
        self.dec2 = archies.dec_archies[self.architags['dec2']]
        
        if 'autoreg' in dtag:
            self.architags['dec3'] = dtag + '_' + str(self.sizes['emb']) + '_' + str(self.sizes['emb']) + '_' + str(self.sizes['vocab'])
            self.dec3 = archies.dec_archies[self.architags['dec3']]
        self.tovocab = nn.Linear(self.sizes['emb'], self.sizes['vocab'], bias=False) 

    def forward(self, z, enc_condition):
        # latentdim x 1 x 1
        dec_gt = self.dec2(F.relu(self.dec_fc(torch.cat((enc_condition, self.dec1(z)), dim=1))))
        # 1 x seqlen x emsize

        bsz = dec_gt.size(0)
        nexchanges = dec_gt.size(1)
        seqlen = dec_gt.size(2)

        if 'autoreg' in self.dtag:
            dec_gt = self.dec3(dec_gt.permute(0,3,1,2))
            dec_gt = self.tovocab(dec_gt.permute(0,2,3,1))
        elif 'standard' in self.dtag:
            dec_gt = self.tovocab(dec_gt)
        
        return dec_gt
