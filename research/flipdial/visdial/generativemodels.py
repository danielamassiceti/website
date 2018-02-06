import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models.vgg as vgg
#import gensim

#import utils
#import net_utils
import decoders
import encoders

PAD_ID = 0
GO_ID = 2
EOS_ID = 1
UNK_ID = 3
 
class imagemodel(nn.Module):
        
    def __init__(self, iarch, learnimgfeats):
        super(imagemodel, self).__init__()
        
        self.iarch = iarch
        self.learnimgfeats = learnimgfeats
        
        fullmodel = vgg.VGG(vgg.make_layers(vgg.cfg['D']))
        fullmodel.load_state_dict(torch.load(self.iarch)) 
        #models.__dict__[iarch](pretrained=True)

        if 'resnet' in self.iarch:
            self.features = nn.Sequential(*list(fullmodel.children())[:-1])
            self.classifier = None
        elif 'vgg' in self.iarch:
            self.features = fullmodel.features
            self.classifier = nn.Sequential(*list(fullmodel.classifier.children())[:-1])
        else:
            print('other image feature networks cannot be handled right now')
            return
        
        if not self.learnimgfeats:
            for p in self.features.parameters():
                p.requires_grad = False

            if self.classifier:
                for p in self.classifier.parameters():
                    p.requires_grad = False
 
    def forward(self, x):
        x = self.features(x)
        if self.classifier: # if using vgg, take l2 norm of features as in VisDial paper
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            l2norm = torch.norm(x, 2, dim=1).view(-1,1)
            x = x.div(l2norm.expand_as(x))
        return x

class dialogmodel(nn.Module):
    def __init__(self, opt, vocabsize):
        super(dialogmodel, self).__init__()
        
        if 'vgg' in opt.iarch:
            self.visfeatsize = 4096
        elif 'resnet' in opt.iarch:
            self.visfeatsize = 512
        self.vocabsize = vocabsize
        self.emsize = opt.emsize
        self.w2v_emsize = opt.w2v_emsize
        self.nhid = opt.nhid
        self.seqlen = opt.seqlen
        self.numexchanges = opt.exchangesperimage*2

        self.embedding = nn.Embedding(vocabsize, self.emsize)
        
        # encoders
        self.input_tag = opt.input_vars
        self.condition_tag = opt.condition_vars

        if opt.fullblock:
            # encoders
            self.encoder = encoders.block_encoder(self.input_tag, self.condition_tag, self.nhid, self.emsize, self.seqlen, self.visfeatsize, 16, self.w2v_emsize, opt.w2vcap)

            # decoders
            self.decoder = decoders.block_decoder(opt.decoder, self.nhid, self.emsize, self.seqlen, 16, self.vocabsize)
            
            if opt.tie_weights:
                self.embedding.weight = self.decoder.tovocab.weight
        else:
            # encoders 
            self.encoder = encoders.item_encoder(self.input_tag, self.condition_tag, self.nhid, self.emsize, self.seqlen, self.visfeatsize, 16, self.w2v_emsize, opt.w2vcap) 
        
            # decoders
            if 'answer' in self.input_tag:
                self.answer_decoder = decoders.item_decoder(opt.decoder, self.nhid, self.emsize, self.seqlen, 16, self.vocabsize)
            if 'question' in self.input_tag:
                self.question_decoder = decoders.item_decoder(opt.decoder, self.nhid, self.emsize, self.seqlen, 16, self.vocabsize)

            if opt.tie_weights:
                if 'answer' in self.input_tag:
                    self.embedding.weight = self.answer_decoder.tovocab.weight
                if 'question' in self.input_tag:
                    self.embedding.weight = self.question_decoder.tovocab.weight
          
    def forward(self, inputs, sampler, fix='none', evaluation=False, decode_mode='recon'):
         
        enc_condition, params = self.encoder(inputs, evaluation)
        nsamples = 8
    
        if decode_mode=='sample':
            params['z'] = []
            for n in xrange(nsamples):
                params['z'].append(sampler(params['mu_p'], params['logvar_p'], evaluation))
        elif decode_mode =='recon':
            params['z'] = sampler(params['mu_q'], params['logvar_q'], evaluation)
        
        if 'dialogblock' in inputs:
            output_dict = {}
            if not fix == 'none': #feed in extra question
                output_dict['tgt'] = self.decoder(params['z'], enc_condition, inputs['gt_emb'], fix)
            else:
                output_dict['tgt'] = self.decoder(params['z'], enc_condition)
        else:
            output=[]
            for n in xrange(nsamples):
                if 'question' in self.input_tag:
                    output.append(self.question_decoder(params['z'][n], enc_condition).squeeze(2).squeeze(2))
                if 'answer' in self.input_tag:
                    output.append(self.answer_decoder(params['z'][n], enc_condition).squeeze(2).squeeze(2))

	params['z'] = torch.stack(params['z'], dim=0)#.squeeze(0)
        output = torch.stack(output, dim=0)#.squeeze(0)
 
        return output, params
    
    def just_encode(self, inputs, evaluation=False):

        enc_condition, params = self.encoder(inputs, evaluation)

        return params 
    
class vismodel(nn.Module):
    def __init__(self, opt):
        super(vismodel, self).__init__()

        if opt.net == 'img_dialog':
            self.model = imagemodel(opt.iarch, opt.learnimgfeats)
        else:
            print 'net argument has to be in (img_dialog)'

        self.net = opt.net

    def forward(self, img):
        return self.model(img) # just use image model
    
class VQAmodel():
    def __init__(self, opt, vocabsize, filename, loadfromfile=False):
        
        self.net = opt.net
        self.learnimgfeats = opt.learnimgfeats 
       
        self.vismodel = vismodel(opt)
        self.dialogmodel = dialogmodel(opt, vocabsize)

        if loadfromfile:
            self.dialogmodel.load_state_dict(torch.load(filename))

        if opt.doubleup:
            self.double()
        
        if opt.gpu > 0:
            self.cuda()
        else:
            self.cpu()
 
    def vismodel(self):
        return self.vismodel
    
    def dialogmodel(self):
        return self.dialogmodel

    def save(self, filename):
        torch.save(self.dialogmodel.cpu().state_dict(), filename)
        self.cuda()    
         
    def cuda(self):
        self.vismodel = self.vismodel.cuda()
        self.dialogmodel = self.dialogmodel.cuda()
        return self
    
    def cpu(self):
        self.vismodel.cpu()
        self.dialogmodel.cpu()
        return self
    
    def double(self):
        self.vismodel.double()
        self.dialogmodel.double()
        return self
    
    def train(self):
        self.vismodel.train()
        self.dialogmodel.train()
        
    def eval(self):
        self.vismodel.eval()
        self.dialogmodel.eval()
