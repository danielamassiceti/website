#import sys, os, time, math, gc, json
#import numpy as np
import torch, os, inflect, pickle
from torch.autograd import Variable
from PIL import Image
import torch.nn as nn
#import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
#import torchvision.datasets as datasets

#user-defined classes
import generativemodels as models
#import losses
import samplers
import utils

################################################################################

class VisDial(object):
    def __init__(self, static_dir):

        # get opts
        self.static_dir = static_dir
        resources_dir = os.path.join(static_dir, 'visdial_resources')
        opt = torch.load(os.path.join(resources_dir, 'genvd_opt.pt'))
        opt.iarch = os.path.join(resources_dir, 'vgg16-397923af.pth')
        opt.gpu = 0 #cpu only
        self.opt = opt
        
        # basic cuda/gpu set-up
        torch.manual_seed(opt.seed)
        if opt.doubleup:
            torch.set_default_tensor_type('torch.DoubleTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
            
        if torch.cuda.is_available():
            if opt.gpu > 0: #gpu
                assert opt.gpu <= torch.cuda.device_count()
                torch.cuda.manual_seed(opt.seed)
                torch.cuda.set_device(opt.gpu-1)
                cudnn.enabled = True
                cudnn.benchmark = True
        
        
        # build vocab
        self.dict = torch.load( os.path.join(resources_dir, 'vocab_visdial_v' + str(opt.datasetversion) + '.pt'))
        ntokens = len(self.dict)
        self.dict.loadwordmodel(os.path.join(resources_dir, 'vocab_visdial_vecs.pt'))
        self.nc = inflect.engine() #convert numbers to text
        
        # build/load model & load data
        self.sampler = samplers.GaussianSampler()
        #criterion = losses.ELBO(sampler=sampler, noprior=opt.noprior, withpos=opt.withpos)
        #if opt.gpu > 0:
        #    criterion = criterion.cuda()
        
        #if not os.path.exists(os.path.join(resultsdir, 'models')):
        #        os.mkdir(os.path.join(resultsdir, 'models'))
        
        #init global buffers
        #globalbuffers.init(opt)
        
        #load trained model
        loadmodelname = 'genvd.pt'
        self.vqamodel = models.VQAmodel(opt, ntokens, os.path.join(resources_dir, loadmodelname), loadfromfile=True)
        self.vqamodel.eval()
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #imagenet
        self.transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ])

        # initialise history
        self.history = Variable(torch.zeros(1,self.opt.exchangesperimage*2,self.opt.seqlen, self.opt.emsize), volatile=True)

    def reset(self, img_path, caption):
        self.set_enc_img(img_path)
        self.set_enc_cap(caption)
        self.history.fill_(0)

    def set_enc_img(self, img_path):
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            
        img = Variable(img.unsqueeze(0), volatile=True)
        self.enc_img = self.vqamodel.vismodel(img)

    def set_enc_cap(self, caption):
        cap_toks = []

        # preprocess caption
        cap_toks = utils.preprocess(caption, self.opt.seqlen, self.dict)

        # get word2vector representation of caption
        cap_toks = torch.LongTensor(cap_toks).view(1,1, self.opt.seqlen)    

        self.enc_cap = Variable(utils.get_w2v(cap_toks, self.dict), volatile=True)

    def check_img(self):
        return isinstance(self.enc_img, Variable)
    
    def check_cap(self):
        return isinstance(self.enc_cap, Variable)

    def get_answer(self, t, question):

        # repack image and caption in Variables
        self.enc_img = Variable( self.enc_img.data, volatile=True)
        self.enc_cap = Variable( self.enc_cap.data, volatile=True)
        self.history = Variable( self.history.data, volatile=True)

        decode_mode = 'sample'
        
        # encode question
        enc_q = torch.LongTensor(utils.preprocess(question, self.opt.seqlen, self.dict))
        enc_q = Variable(enc_q, volatile=True)
        enc_q = self.vqamodel.dialogmodel.embedding(enc_q.view(1, -1)).view(1,1,self.opt.seqlen, self.opt.emsize)

        # predict answer
        output, params = self.vqamodel.dialogmodel({'img': self.enc_img, 'cap': self.enc_cap, 'question': enc_q, 'history': self.history}, self.sampler, evaluation=True, decode_mode=decode_mode)
        #ce_loss, kld_loss = criterion(output_dict, extract_by_t(gt,tgt_inds), params)
        logpdf_scores = self.sampler.logpdf(params['z'], params['mu_p'], params['logvar_p']) 
        sorted_scores, indices = torch.sort(logpdf_scores.squeeze(), descending=True) 
        output = output.index_select(0, indices)
        
        # copy question and answer into history
        self.history.data.narrow(1, t, 1).copy_(enc_q.data)
        #for n in xrange(output_dict['tgt'].size(0)): #num samples
        a_idx = utils.getidx_and_pad(output[0])
        enc_a = self.vqamodel.dialogmodel.embedding(a_idx.view(1,-1)).view(1,1,self.opt.seqlen, self.opt.emsize)
        self.history.data.narrow(1, t+1, 1).copy_(enc_a.data)

        return utils.formatoutput({'tgt':output.squeeze(1)}, self.dict, self.opt.seqlen)['tgt']
        
        
        #emb_gt = self.vqamodel.dialogmodel.embedding(gt['tgt'].view(bsz, -1)).view(bsz, -1, opt.seqlen, opt.emsize)
        #       q_inds = torch.arange(0,opt.exchangesperimage*2,2).type_as(gt['tgt'].data).long().view(-1,1)
 
        #for si,ss in enumerate(decode_modes):
        #        
        #    if opt.fulleval:
        #                emb_dhat = emb_gt.clone()
        #            all_inds = torch.arange(0,opt.exchangesperimage*2).type_as(gt['tgt'].data).long().view(-1,1)

        #            foutput = []

        #            t1 = time.time()
        #            for t, q_ind in enumerate(q_inds):
        #            
        #                t2 = time.time()
        #            
        #                if 'answer' in opt.input_vars and 'question' in opt.input_vars:
        #                    tgt_inds = Variable(torch.cat((q_ind, q_ind+1)), volatile=True)
        #                elif 'answer' in opt.input_vars:
        #                    tgt_inds = Variable(q_ind+1, volatile=True)
        #                elif 'question' in opt.input_vars:
        #                    tgt_inds = Variable(q_ind, volatile=True)
 
        #                ce_loss, kld_loss = criterion(output_dict, extract_by_t(gt,tgt_inds), params)
        #                numsamples = output_dict['tgt'].size(0)*output_dict['tgt'].size(1)
        #                loss = (ce_loss + kld_loss)/numsamples
        #                #p_x_given_zc = -1*ce_loss + p_z_given_c
        #                t3 = time.time()
        #            
        #                if opt.withrank:
        #                    res1, res2, res3 = utils.rankcandidates(output_dict['tgt'][:,-1].unsqueeze(1), targets['answer_options'][:,t].unsqueeze(1), targets['gtidxs'][:,t].unsqueeze(1), dictionary) 
        #                    log.info('{} t={}\t#quests: {}\tr@1: {:.2f}\tr@5: {:.2f}\tr@10: {:.2f}\tMR: {:.2f}\tMRR: {:.4f}'.format(ss, t, bsz, res2[0], res2[1], res2[2], res1, res3))
        #
        #                    mrank[ss].update(res1)
        #                    for k in xrange(len(res2)):
        #                        recallatk[ss][k].update(res2[k])
        #                    mrrs[ss].update(res3)
        #                
        #                if opt.fulleval:
        #                    if 'question' in opt.input_vars:
        #                        cap_q_sim = utils.caption_question_sim(output_dict['tgt'][:,0].unsqueeze(1), targets['cap_idx'], dictionary)
        #                        cqsim[ss].update(cap_q_sim)
        #                    new_embs = net.dialogmodel.embedding(utils.getidx_and_pad(output_dict['tgt']).view(bsz,-1)).view(bsz,-1,opt.seqlen, opt.emsize)
        #                    for ti,tgt_ind in enumerate(tgt_inds):
        #                        emb_dhat[:,tgt_ind.data[0]] = new_embs.index_select(1, Variable(torch.Tensor([ti]).type_as(tgt_inds.data))) 
        #                    dhat_params = net.dialogmodel.just_encode({'img': img, 'cap': targets['cap'], 'question': emb_gt.index_select(1,Variable(q_ind, volatile=True)), 'history': emb_dhat * Variable(mask, volatile=True), 'answer': emb_dhat.index_select(1,Variable(q_ind+1,volatile=True))}, evaluation=True)
        #                    Dz_sim = sampler.kld({'mu_q': dhat_params['mu_q'] , 'logvar_q': dhat_params['logvar_q'], 'mu_p': params['mu_q'] , 'logvar_p': params['logvar_q']})
        #                    Dzsim[ss].update(Dz_sim.data[0]/numsamples)
        #                    del new_embs
        #             
        #                ce_losses[ss].update(ce_loss.data[0]/numsamples)
        #                kld_losses[ss].update(kld_loss.data[0]/numsamples)
        #                losses[ss].update(loss.data[0])
        #                #likelihoods[ss].update(p_x_given_zc.data[0]/numsamples)
        #                
        #                mask.index_fill_(1, all_inds[all_inds <= q_ind+1].squeeze(), 1)
        #     
        #                if save or opt.verbose:
        #                    if si==0:
        #                        save_dict = extract_by_t(gt, tgt_inds)
        #                    else:
        #                        save_dict = {}
        #                    if i==0 or i==10 or i==20 or i==30:
        #                        save_dict['dec'] = output_dict.get('tgt')
        #                        #if opt.sampleprior:
        #                            #save_dict['gen'] = gen_gt_t
        #                        #if opt.withpos:
        #                        #    save_dict['pos'] = output_dict['pos']
        #                        foutput.append(utils.formatoutput(save_dict, dictionary, opt.seqlen))
        #                    del save_dict

        #            t4 = time.time()
        #        
        #return losses['sample']

def generate_block(self, f):
        inputs = {'img': img, 'dialogblock': emb_gt, 'cap': targets['cap']}

        t1 = time.time()
        output_dict, params = net.dialogmodel(inputs, sampler, opt.fix, evaluation=True, decode_mode=ss)
        t2 = time.time()
 
        ce_loss, kld_loss = criterion(output_dict, gt, params)
        numsamples = output_dict['tgt'].size(0)*output_dict['tgt'].size(1)
        loss = (ce_loss + kld_loss)/numsamples
        #p_x_given_zc = -1*ce_loss + p_z_given_c
        t3 = time.time()
        
        if opt.withrank:
            a_inds = Variable(torch.arange(1,output_dict['tgt'].size(1),2).type_as(targets['answer_options'].data), volatile=True)
            res1, res2, res3 = utils.rankcandidates(output_dict['tgt'].index_select(1, a_inds), targets['answer_options'], targets['gtidxs'], dictionary) 
            log.info(ss + ';  #quests: {}\tr@1: {:.2f}\tr@5: {:.2f}\tr@10: {:.2f}\tMR: {:.2f}\tMRR: {:.4f}'.format(bsz, res2[0], res2[1], res2[2], res1, res3))
        
            mrank[ss].update(res1)
            for k in xrange(len(res2)):
                recallatk[ss][k].update(res2[k])
            mrrs[ss].update(res3)

        if opt.fulleval:
            cap_q_sim = utils.caption_question_sim(output_dict['tgt'].index_select(1,Variable(q_inds.squeeze(),volatile=True)), targets['cap_idx'], dictionary)
            cqsim[ss].update(cap_q_sim)

            emb_dhat = net.dialogmodel.embedding(utils.getidx_and_pad(output_dict['tgt']).view(bsz,-1)).view(bsz,-1,opt.seqlen, opt.emsize)
            dhat_params = net.dialogmodel.just_encode({'img': img, 'dialogblock': emb_dhat, 'cap': targets['cap']}, evaluation=True)
            Dz_sim = sampler.kld({'mu_q': dhat_params['mu_q'] , 'logvar_q': dhat_params['logvar_q'], 'mu_p': params['mu_q'] , 'logvar_p': params['logvar_q']})
            Dzsim[ss].update(Dz_sim.data[0]/numsamples)
        
        ce_losses[ss].update(ce_loss.data[0]/numsamples)
        kld_losses[ss].update(kld_loss.data[0]/numsamples)
        losses[ss].update(loss.data[0])
        #likelihoods[ss].update(p_x_given_zc.data[0]/numsamples)
        
        if save or opt.verbose: 
            foutput = {}
            save_dict = {}
            if si == 0:
                save_dict['tgt'] = gt.get('tgt')
            if i==0 or i==10 or i==20 or i==30:
                save_dict['dec'] = output_dict.get('tgt')
                foutput = utils.formatoutput(save_dict, dictionary, opt.seqlen)
            del save_dict

            for (k,v) in foutput.iteritems():
                for bb in xrange(len(v)):
                    ex=0
                    if save and k in writefiles[ss]:
                        writefiles[ss][k].write(targets['imgnames'][bb])
                    elif opt.verbose:
                        print targets['imgnames'][bb]
                    for nnn in xrange(len(v[bb])):
                        ex+=1
                        if save and k in writefiles[ss]:
                            writefiles[ss][k].write(str(ex) + ':' + v[bb][nnn])
                        elif opt.verbose:
                            print k + '-' + str(ex) + ': ' + v[bb][nnn]

        t4 = time.time()
            
#test_loss = net_tils.evaluate(vqamodel, test_img, data.prepare_batch, criterion, sampler, dict, opt, epoch, log, testset, 1) 
    

