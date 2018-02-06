import os, time, math
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import globalbuffers
import utils
import gc
from numpy import random

def getvisfeatsize(net, iarch, sgfeatsize):
    
    if 'img' in net:
        if iarch.startswith('vgg'):
            imgfeatsize = 4096
        elif iarch.startswith('resnet'):
            imgfeatsize = 512
        else:
            assert False, 'visfeatsize unknown for this image model architecture'
    if net == 'dialog':
        visfeatsize = 0
    elif net == 'img_dialog':
        visfeatsize = imgfeatsize
    elif net == 'sgbox_dialog' or net == 'sgbox_sgrel_dialog' or net == 'sgrel_dialog' or net == 'img_sgbox_dialog' or net == 'img_sgbox_sgrel_dialog': 
        visfeatsize = sgfeatsize

    return visfeatsize

def getcombinedfeatsize(iarch, sgfeatsize):
    
    if iarch.startswith('vgg'):
        imgfeatsize = 4096
    elif iarch.startswith('resnet'):
        imgfeatsize = 512
    else:
        assert False, 'visfeatsize unknown for this image model architecture'
    
    return imgfeatsize + sgfeatsize

def getvalidmask(vertvalid, edgevalid, net):
    if 'sgbox' in net and 'sgrel' in net:
        validmask = vertvalid * edgevalid
        return validmask, int(validmask.data.sum())
    elif 'sgbox' in net:
        validmask = vertvalid
        return validmask, int(validmask.data.sum())
    elif 'sgrel' in net:
        validmask = edgevalid
        return validmask, int(validmask.data.sum())

def prune_by_bsz(gt, context, bsz):
    for k,v in gt.iteritems():
        gt[k] = v[0:bsz]
    for k,v in context.iteritems():
        if isinstance(v, Variable):
            context[k] = v[0:bsz]
    return gt, context

def extract_by_t(gt, t_inds):
    gt_t = {}

    for k,v in gt.iteritems():
        gt_t[k] = v.index_select(1, t_inds)

    return gt_t

def get_klanneal_weight(epoch, max_klanneal_epoch=50, anneal_type='linear'):
    klannealweight = 1
    if anneal_type == 'linear':
        if epoch < max_klanneal_epoch:
            klannealweight = torch.linspace(0, 1, max_klanneal_epoch)[epoch]
    elif anneal_type == 'log':
        if epoch <= max_klanneal_epoch:
            klannealweight = torch.logspace(-5, 0, max_klanneal_epoch)[epoch-1]

    return klannealweight

def evaluate(net, data_loader, prepare_batch_fn, criterion, sampler, dictionary, opt, ep, log, set='test', save=0):
    
    resultsdir = os.path.join(opt.resultsdir, 'experiment_id' + str(opt.id))
    logfilename = opt.modelname
       
    decode_modes = ['sample', 'recon']
    
    writefiles,writefilenames,batch_time,ce_losses,kld_losses,losses,mrank,recallatk,mrrs,likelihoods,cqsim,Dzsim = {},{},{},{},{},{},{},{},{},{},{},{}
    if not os.path.exists(os.path.join(resultsdir, 'preds')):
        os.mkdir(os.path.join(resultsdir, 'preds'))
     
    for ss in decode_modes:
        writefiles[ss] = {}
        writefilenames[ss] = {}
        writefilenames['sample']['tgt'] = os.path.join(resultsdir, 'preds', logfilename + '_' + set + '_gt_tgt.txt')
        writefilenames[ss]['dec'] = os.path.join(resultsdir, 'preds', logfilename + '_' + set + '_' + ss + '_pred_epoch' + str(ep) + '.txt')
        if opt.interp_latents:
            writefilenames[ss]['interp'] = os.path.join(resultsdir, 'preds', logfilename + '_' + set + '_interp_epoch' + str(ep) + '.txt')
        
        for k,v in writefilenames[ss].iteritems():
            if not os.path.exists(v):
                writefiles[ss][k] = open(v, 'w')
     
        batch_time[ss] = utils.AverageMeter()
        ce_losses[ss] = utils.AverageMeter()
        kld_losses[ss] = utils.AverageMeter()
        losses[ss] = utils.AverageMeter()
        mrank[ss] = utils.AverageMeter()
        recallatk[ss] = [utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter()] # for k=1, 5 and 10
        mrrs[ss] = utils.AverageMeter()
        likelihoods[ss] = utils.AverageMeter()
        cqsim[ss] = utils.AverageMeter()
        Dzsim[ss] = utils.AverageMeter()
    
    net.eval()
    
    end = time.time()
    for i, ((b_imgs,_), b_targets) in enumerate(data_loader):

        bsz = b_targets[1].size(0)

        #log.info('processing batch ' + str(i+1) + '/' + str(len(data_loader)))

        gt = {}
        context = {}
    
        if opt.dataset == 'visdial':
            # wraps everything as Variables
            if opt.w2vcap:
                img, targets = prepare_batch_fn(b_imgs, b_targets, opt.exchangesperimage, opt.seqlen, dictionary, withcuda=opt.gpu>0, evaluation=True)
            else:
                img, targets = prepare_batch_fn(b_imgs, b_targets, opt.exchangesperimage, opt.seqlen, withcuda=opt.gpu>0, evaluation=True)
            gt['tgt'] = targets['dialog']
            if not opt.savedimgfeats:
                img = net.vismodel(img)

        gt, context = prune_by_bsz(gt, context, bsz)
        
        # update global buffers with question, answer, caption lengths
        #gt_lens = utils.updatelengths(gt)

        emb_gt = net.dialogmodel.embedding(gt['tgt'].view(bsz, -1)).view(bsz, -1, opt.seqlen, opt.emsize)
        q_inds = torch.arange(0,opt.exchangesperimage*2,2).type_as(gt['tgt'].data).long().view(-1,1)
        all_inds = torch.arange(0,opt.exchangesperimage*2).type_as(gt['tgt'].data).long().view(-1,1)
 
        for si,ss in enumerate(decode_modes):
            
            if opt.blockmode == 'full':
                t1 = time.time()
                t2 = time.time()
                 
                inputs = {'img':img, 'dialogblock': emb_gt, 'cap': targets['cap']}
                output_dict, params = net.dialogmodel(inputs, sampler, opt.fix, evaluation=True, decode_mode=ss)
 
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
            
            elif opt.blockmode == 'iterative':
            
                output_dict = {}
                pad_emb = net.dialogmodel.embedding(Variable(gt['tgt'].data.new(1).fill_(dictionary.word2idx['PAD']), volatile=evaluation))
                eos_emb = net.dialogmodel.embedding(Variable(gt['tgt'].data.new(1).fill_(dictionary.word2idx['EOS']), volatile=evaluation))
                emb_gt_padfill = emb_gt.clone()
                emb_gt_padfill.data.copy_(pad_emb.data.view(1,1,1,-1).expand(bsz, gt['tgt'].size(1), opt.seqlen, opt.emsize))
                emb_gt_padfill.data[:,:,0] = eos_emb.data.view(1,1,1,-1).expand(bsz, gt['tgt'].size(1), 1, opt.emsize)
                output_dict['tgt'] = iterative_block_forward(net, img, gt, targets, dictionary, sampler, opt, ss, evaluation)

            elif opt.blockmode == 'none':
                if 'item' in opt.input_vars: #  encode item at each t
                    t_inds = torch.arange(0,opt.exchangesperimage*2).type_as(gt['tgt'].data).long().view(-1,1)
                else: # encode only answers  
                    t_inds = torch.arange(1,opt.exchangesperimage*2,2).type_as(gt['tgt'].data).long().view(-1,1)
                mask = torch.zeros(bsz, opt.exchangesperimage*2, opt.seqlen, opt.emsize).type_as(gt['tgt'].data).float().fill_(0)
                
                predicted_dialogue = gt['tgt'].data.clone()
                emb_history = net.dialogmodel.embedding(Variable(predicted_dialogue, volatile=True).view(bsz,-1)).view(bsz, -1, opt.seqlen, opt.emsize)
                
                foutput = []

                for t, t_ind in enumerate(t_inds):
                
                    t1 = time.time()
                    if opt.usepreds: # if using predictions, predicted_dialogue is updated, and embedding must be updated
                        emb_history = net.dialogmodel.embedding(Variable(predicted_dialogue, volatile=True).view(bsz,-1)).view(bsz, -1, opt.seqlen, opt.emsize)
                    if 'item' in opt.input_vars:
                        emb_item = net.dialogmodel.embedding(gt['tgt'].index_select(1,Variable(t_ind, volatile=True)).view(bsz,-1)).view(bsz, -1, opt.seqlen, opt.emsize)
                        output_dict, params = net.dialogmodel({'img': img, 'cap': targets['cap'], 'item': emb_item, 'history': emb_history * Variable(mask, volatile=True)}, sampler, evaluation=True, decode_mode=ss)
                    else:
                        emb_question = net.dialogmodel.embedding(gt['tgt'].index_select(1,Variable(t_ind-1, volatile=True)).view(bsz,-1)).view(bsz, -1, opt.seqlen, opt.emsize)
                        emb_answer = net.dialogmodel.embedding(gt['tgt'].index_select(1,Variable(t_ind, volatile=True)).view(bsz,-1)).view(bsz, -1, opt.seqlen, opt.emsize)
                        output_dict, params = net.dialogmodel({'img': img, 'cap': targets['cap'], 'question': emb_question, 'history': emb_history * Variable(mask, volatile=True), 'answer': emb_answer}, sampler, evaluation=True, decode_mode=ss)
            
                    t2 = time.time()
                
                    ce_loss, kld_loss = criterion(output_dict, extract_by_t(gt,Variable(t_ind)), params)
                    numsamples = output_dict['tgt'].size(0)*output_dict['tgt'].size(1)
                    loss = (ce_loss + kld_loss)/numsamples
                    #p_x_given_zc = -1*ce_loss + p_z_given_c
                    t3 = time.time()
                
                    #if opt.interp_latents and writefiles['interp']:
                    #    interp = net.dialogmodel.interp_latents(opt.interp_steps, evaluation=True)
                    #    finterp = utils.formatoutput(interp, dictionary, opt.seqlen)
                    #    writefiles['interp'].write('-'*10 + '\n')
                    #    for fi in xrange(len(finterp['interp'])):
                    #        writefiles['interp'].write(str(fi) + ': ' + finterp['interp'][fi])
                
                    if opt.withrank:
                        if 'item' in opt.input_vars:
                            if not t % 2: #if not a question
                                res1, res2, res3 = utils.rankcandidates(output_dict['tgt'], targets['answer_options'][:,(t-1)/2].unsqueeze(1), targets['gtidxs'][:,(t-1)/2].unsqueeze(1), dictionary) 
                        else:
                            res1, res2, res3 = utils.rankcandidates(output_dict['tgt'], targets['answer_options'][:,t].unsqueeze(1), targets['gtidxs'][:,t].unsqueeze(1), dictionary) 
                        log.info('{} t={}\t#quests: {}\tr@1: {:.2f}\tr@5: {:.2f}\tr@10: {:.2f}\tMR: {:.2f}\tMRR: {:.4f}'.format(ss, t, bsz, res2[0], res2[1], res2[2], res1, res3))
    
                        mrank[ss].update(res1)
                        for k in xrange(len(res2)):
                            recallatk[ss][k].update(res2[k])
                        mrrs[ss].update(res3)
                    
                    if opt.fulleval:
                        if 'item' in opt.input_vars:
                            if t % 2:
                                cap_q_sim = utils.caption_question_sim(output_dict['tgt'], targets['cap_idx'], dictionary)
                                cqsim[ss].update(cap_q_sim)
                        new_emb_item = net.dialogmodel.embedding(utils.getidx_and_pad(output_dict['tgt']).view(bsz,-1)).view(bsz,-1,opt.seqlen, opt.emsize)
                        if 'item' in opt.input_vars:
                            dhat_params = net.dialogmodel.just_encode({'img': img, 'cap': targets['cap'], 'item': new_emb_item, 'history': emb_history * Variable(mask, volatile=True)}, evaluation=True)
                        else:
                            dhat_params = net.dialogmodel.just_encode({'img': img, 'cap': targets['cap'], 'question': emb_question, 'history': emb_history * Variable(mask,
                                volatile=True), 'answer': new_emb_item}, evaluation=True)

                        Dz_sim = sampler.kld({'mu_q': dhat_params['mu_q'] , 'logvar_q': dhat_params['logvar_q'], 'mu_p': params['mu_q'] , 'logvar_p': params['logvar_q']})
                        Dzsim[ss].update(Dz_sim.data[0]/numsamples)
                        del new_emb_item
                        del dhat_params 
                 
                    ce_losses[ss].update(ce_loss.data[0]/numsamples)
                    kld_losses[ss].update(kld_loss.data[0]/numsamples)
                    losses[ss].update(loss.data[0])
                    #likelihoods[ss].update(p_x_given_zc.data[0]/numsamples)
                    
                    mask.index_fill_(1, all_inds[all_inds <= t_ind].squeeze(), 1)
         
                    if save or opt.verbose:
                        if si==0:
                            save_dict = extract_by_t(gt, Variable(t_ind))
                        else:
                            save_dict = {}
                        if i==0 or i==10 or i==20 or i==30:
                            save_dict['dec'] = output_dict.get('tgt')
                            #if opt.sampleprior:
                                #save_dict['gen'] = gen_gt_t
                            #if opt.withpos:
                            #    save_dict['pos'] = output_dict['pos']
                            foutput.append(utils.formatoutput(save_dict, dictionary, opt.seqlen))
                        del save_dict

                if save or opt.verbose:
                    for bb in xrange(bsz):
                        ex=0
                        for dialog_item in foutput:
                            ex+=1
                            for (k,v) in dialog_item.iteritems():
                                if save and k in writefiles[ss]:
                                    if ex==1:
                                        writefiles[ss][k].write(targets['imgnames'][bb])
                                    writefiles[ss][k].write(str(ex) + ':' + v[bb][0])
                                elif opt.verbose:
                                    if ex==1:
                                        print targets['imgnames'][bb]
                                    print k + ': ' + v[bb][0]
                
                t4 = time.time()
            
            log.info('{0}_eval: [{1}][{2}/{3}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:8.4f} ({loss.avg:8.4f})\t'
                'CEL {cel.val:8.4f} ({cel.avg:8.4f})\t'
                'KLD {kld.val:8.4f} ({kld.avg:8.4f})\t'
                'LHood {lhood.val:8.4f} ({lhood.avg:8.4f})\t'
                'MR {mr.val:.2f} ({mr.avg:.2f})\t'
                'r@1 {rat1.val:.5f} ({rat1.avg:.5f})\t'
                'r@5 {rat5.val:.5f} ({rat5.avg:.5f})\t'
                'r@10 {rat10.val:.5f} ({rat10.avg:.5f})\t'
                'MRR {mrr.val:.5f} ({mrr.avg:.5f})'.format(ss, ep, i+1, len(data_loader),
                    batch_time=batch_time[ss], loss=losses[ss], cel=ce_losses[ss], kld=kld_losses[ss], lhood=likelihoods[ss],
                    mr=mrank[ss], rat1=recallatk[ss][0], rat5=recallatk[ss][1], rat10=recallatk[ss][2], mrr=mrrs[ss]))
            if opt.fulleval:
                log.info('{0}_eval: [{1}][{2}/{3}]\t'
                    'cap_q_sim {capq.val:4.4f} ({capq.avg:4.4f})\t'
                    'D_z_sim {dz.val:8.4f} ({dz.avg:8.4f})'.format(ss, ep, i+1, len(data_loader), capq=cqsim[ss], dz=Dzsim[ss]))

            
            del output_dict
            del params
            del ce_loss
            del kld_loss
            #del p_z_given_c

     
        del gt
        del context
        del img
        del targets
        del emb_gt

    for ss in decode_modes: 
        if save:
            for k,v in writefiles[ss].iteritems():
                if v:
                    v.close()

        celossesfilename = os.path.join(resultsdir, 'logs', logfilename + '_epoch' + str(ep) + '_' + set + '_' + ss + 'fillin_celoss.meter')
        if not os.path.exists(celossesfilename):
            torch.save(ce_losses[ss], celossesfilename)
        kldlossesfilename = os.path.join(resultsdir, 'logs', logfilename + '_epoch' + str(ep) + '_' + set + '_' + ss + 'fillin_kldloss.meter')
        if not os.path.exists(kldlossesfilename):
            torch.save(kld_losses[ss], kldlossesfilename)
        samplelossesfilename = os.path.join(resultsdir, 'logs', logfilename + '_epoch' + str(ep) + '_' + set + '_' + ss + 'fillin_loss.meter')
        if not os.path.exists(samplelossesfilename):
            torch.save(losses[ss], samplelossesfilename)
        likelihoodfilename = os.path.join(resultsdir, 'logs', logfilename + '_epoch' + str(ep) + '_' + set + '_' + ss + 'fillin_likelihood.meter')
        if not os.path.exists(likelihoodfilename):
            torch.save(likelihoods[ss], likelihoodfilename)
        mrankfilename = os.path.join(resultsdir, 'logs', logfilename + '_epoch' + str(ep) + '_' + set + '_' + ss + 'fillin_mrank.meter')
        recallatkfilename = os.path.join(resultsdir, 'logs', logfilename + '_epoch' + str(ep) + '_' + set + '_' + ss + 'fillin_recallatk.meter')
        mrrfilename = os.path.join(resultsdir, 'logs', logfilename + '_epoch' + str(ep) + '_' + set + '_' + ss + 'fillin_mrr.meter')
        if opt.withrank:
            if not os.path.exists(mrankfilename):
                torch.save(mrank[ss], mrankfilename)
            if not os.path.exists(recallatkfilename):
                torch.save(recallatk[ss], recallatkfilename)
            if not os.path.exists(mrrfilename):
                torch.save(mrrs[ss], mrrfilename)
        if opt.fulleval:
            capquestsimfilename = os.path.join(resultsdir, 'logs', logfilename + '_epoch' + str(ep) + '_' + set + '_' + ss + 'fillin_capquestsim.meter')
            if not os.path.exists(capquestsimfilename):
                torch.save(cqsim[ss], capquestsimfilename)
            Dzsimfilename = os.path.join(resultsdir, 'logs', logfilename + '_epoch' + str(ep) + '_' + set + '_' + ss + 'fillin_Dzsim.meter')
            if not os.path.exists(Dzsimfilename):
                torch.save(Dzsim[ss], Dzsimfilename)
     
    return losses['sample']

def iterative_block_forward(net, emb_gt, t, img, gt, targets, dictionary, sampler, opt, ss, evaluation):
    bsz = gt['tgt'].size(0)
    
    emb_gt = net.dialogmodel.embedding(gt['tgt'].view(bsz, -1)).view(bsz, -1, opt.seqlen, opt.emsize)
    q_inds = torch.arange(0,opt.exchangesperimage*2,2).type_as(gt['tgt'].data).long().view(-1,1)
    all_inds = torch.arange(0,opt.exchangesperimage*2).type_as(gt['tgt'].data).long().view(-1,1)
    
    filled_out = []

    pad_emb = net.dialogmodel.embedding(Variable(gt['tgt'].data.new(1).fill_(dictionary.word2idx['PAD']), volatile=evaluation))
    eos_emb = net.dialogmodel.embedding(Variable(gt['tgt'].data.new(1).fill_(dictionary.word2idx['EOS']), volatile=evaluation))
    emb_gt_padfill = emb_gt.clone()
    emb_gt_padfill.data.copy_(pad_emb.data.view(1,1,1,-1).expand(bsz, gt['tgt'].size(1), opt.seqlen, opt.emsize))
    emb_gt_padfill.data[:,:,0] = eos_emb.data.view(1,1,1,-1).expand(bsz, gt['tgt'].size(1), 1, opt.emsize)
    if opt.usepreds:
        if opt.pred == 'a':
            for t, q_ind in enumerate(q_inds):
    	        emb_gt_padfill[:,q_ind] = emb_gt[:,q_ind]
                inputs = {'img': img, 'dialogblock': emb_gt_padfill, 'cap': targets['cap']}
                output_t_dict, params = net.dialogmodel(inputs, sampler, opt.fix, evaluation=evaluation, decode_mode=ss)
                filled_out.append(output_t_dict['tgt'][:,q_ind[0]:q_ind[0]+2])
                emb_gt_padfill[:,q_ind[0]+1] = emb_gt[:,q_ind[0]+1]
            filled_out = torch.stack(filled_out, 1)
    	    filled_out = filled_out.view(bsz, -1, opt.seqlen, filled_out.size(4))

        elif opt.pred == 'qa':
            for t, t_ind in enumerate(all_inds):
                inputs = {'img': img, 'dialogblock': emb_gt_padfill, 'cap': targets['cap']}
                output_t_dict, params = net.dialogmodel(inputs, sampler, opt.fix, evaluation=evaluation, decode_mode=ss)
                filled_out.append(output_t_dict['tgt'][:,t])
                emb_gt_padfill[:,t] = emb_gt[:,t]
            filled_out = torch.stack(filled_out, 1)
    else:
        if opt.pred == 'a':
            for t, q_ind in enumerate(q_inds):
    	        emb_gt_padfill[:,q_ind] = emb_gt[:,q_ind]
                inputs = {'img': img, 'dialogblock': emb_gt_padfill, 'cap': targets['cap']}
                output_t_dict, params = net.dialogmodel(inputs, sampler, opt.fix, evaluation=evaluation, decode_mode=ss)
                filled_out.append(output_t_dict['tgt'][:,q_ind[0]:q_ind[0]+2])
    	        maxval, maxidx = torch.max(output_t_dict['tgt'][:,q_ind[0]+1], dim=2)
    	        for b_zz in xrange(maxidx.size(0)):
    	            eos_pos = maxidx[b_zz].eq(1).nonzero()
                    if eos_pos.size():
     	                eos_to_end = range(eos_pos.data[0][0]+1, maxidx.size(1))
    	                maxidx[b_zz][eos_to_end].data = torch.Tensor(1).fill_(dictionary.word2idx['PAD']).expand(len(eos_to_end)).type_as(gt['tgt'].data)
                emb_gt_padfill[:,q_ind[0]+1].data = net.dialogmodel.embedding(maxidx.view(-1)).view(bsz, opt.seqlen, opt.emsize).unsqueeze(1).data
            filled_out = torch.stack(filled_out, 1)
            filled_out = filled_out.view(bsz, -1, opt.seqlen, filled_out.size(4))
        elif opt.pred == 'qa':
            for t, t_ind in enumerate(all_inds):
                inputs = {'img': img, 'dialogblock': emb_gt_padfill, 'cap': targets['cap']}
                output_t_dict, params = net.dialogmodel(inputs, sampler, opt.fix, evaluation=evaluation, decode_mode=ss)
                filled_out.append(output_t_dict['tgt'][:,t])
    	        maxval, maxidx = torch.max(output_t_dict['tgt'][:,t], dim=2)
    	        for b_zz in xrange(maxidx.size(0)):
    	            eos_pos = maxidx[b_zz].eq(1).nonzero()
                    if eos_pos.size():
     	                eos_to_end = range(eos_pos.data[0][0]+1, maxidx.size(1))
    	                maxidx[b_zz][eos_to_end].data = torch.Tensor(1).fill_(dictionary.word2idx['PAD']).expand(len(eos_to_end)).type_as(gt['tgt'].data)
                emb_gt_padfill[:,t].data = net.dialogmodel.embedding(maxidx.view(-1)).view(bsz, opt.seqlen, opt.emsize).unsqueeze(1).data
            filled_out = torch.stack(filled_out, 1)
    del emb_gt_padfill

    return filled_out

def train(net, data_loader, prepare_batch_fn, criterion, sampler, dictionary, opt, ep, log):

    resultsdir = os.path.join(opt.resultsdir, 'experiment_id' + str(opt.id))
    logfilename = opt.modelname

    klannealweight = get_klanneal_weight(ep, opt.max_klanneal_epoch, opt.klanneal_type)

    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    ce_losses = utils.AverageMeter()
    kld_losses = utils.AverageMeter()
    losses = utils.AverageMeter()
    likelihoods = utils.AverageMeter()
    grads = utils.AverageMeter()
     
    net.train()
    bn_c = 0
    for mod in net.dialogmodel.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            bn_c += 1
            mod.momentum = 0.001
    log.info(str(bn_c) + ' BatchNorm2d layers updated to momentum=0.001!')

     
    for i, ((b_imgs,_), b_targets) in enumerate(data_loader):

        end = time.time()
   	bsz = b_targets[1].size(0)
        gt = {}
        context = {}

        if bsz < opt.batch_size:
            log.info('Last batch too small - BatchNorm2d fixed')
            for mod in net.dialogmodel.modules():
                if isinstance(mod, torch.nn.BatchNorm2d):
                    mod.eval()

        if globalbuffers.vis_optimizer: #if no learnable parameters in vismodel
            globalbuffers.vis_optimizer.zero_grad()
        
        if opt.dataset == 'visdial':
            # wraps everything as Variables
            if opt.w2vcap:
                img, targets = prepare_batch_fn(b_imgs, b_targets, opt.exchangesperimage, opt.seqlen, dictionary, withcuda=opt.gpu>0) 
            else:
                img, targets = prepare_batch_fn(b_imgs, b_targets, opt.exchangesperimage, opt.seqlen, withcuda=opt.gpu>0) 
            gt['tgt'] = targets['dialog']
            if not opt.savedimgfeats:
                img = net.vismodel(img)
        
        gt, context = prune_by_bsz(gt, context, bsz)
        #gt_lens = utils.updatelengths(gt) 
        
        data_time.update(time.time() - end)   
             
        if opt.blockmode == 'full':
            loss = 0
            globalbuffers.dialog_optimizer.zero_grad()
            
            t1 = time.time()
            emb_gt = net.dialogmodel.embedding(gt['tgt'].view(bsz,-1)).view(bsz, -1, opt.seqlen, opt.emsize)
            output_dict, params = net.dialogmodel({'img': img, 'dialogblock' : emb_gt, 'cap': targets['cap']}, sampler)
            t2 = time.time()
            
            ce_loss, kld_loss = criterion(output_dict, gt, params)
            numsamples = output_dict['tgt'].size(0)*output_dict['tgt'].size(1)
            loss = (ce_loss + (klannealweight * kld_loss))/numsamples
            #p_x_given_zc = -1*ce_loss + p_z_given_c
            t3 = time.time()
              
            loss.backward() #must be true if there are parameters in vis_optimiser 
            #torch.nn.utils.clip_grad_norm(net.dialogmodel.parameters(), opt.clip)
            t4 = time.time()

            globalbuffers.dialog_optimizer.step()
            del emb_gt
            t5 = time.time()
            #for mod in net.dialogmodel.children():
            #    if isinstance(mod, torch.nn.BatchNorm2d):
            #        if np.isnan(mod.running_mean.sum()) or np.isnan(mod.running_var.sum()):
            #            print 'BAD'
             
            ce_losses.update(ce_loss.data[0]/numsamples)
            kld_losses.update(kld_loss.data[0]/numsamples)
            losses.update(loss.data[0])
            #likelihoods.update(p_x_given_zc.data[0]/numsamples)
            grads.update(utils.sumParamGrads([net.vismodel, net.dialogmodel])/numsamples)
            #print ('t == encode/decode: ' + str(t2-t1) + '; compute loss: ' + str(t3-t2) + '; back: ' + str(t4-t3) + '; step: ' + str(t5-t4))
        
        elif opt.blockmode == 'iterative' :
            
            all_inds = torch.arange(0,opt.exchangesperimage*2).type_as(gt['tgt'].data).long()
            if 'answer' in opt.input_vars and 'question' in opt.input_vars:
                t_inds = torch.arange(0,opt.exchangesperimage*2).type_as(gt['tgt'].data).long().view(-1,1)
            elif 'answer' in opt.input_vars:
                t_inds = torch.arange(1,opt.exchangesperimage*2,2).type_as(gt['tgt'].data).long().view(-1,1)
            
            masked_gt = torch.Tensor(bsz, opt.exchangesperimage*2,opt.seqlen).type_as(gt['tgt'].data).fill_(dictionary.word2idx['PAD'])
            masked_gt[:,:,0].fill_(dictionary.word2idx['EOS'])

            for t, t_ind in enumerate(t_inds):
            
                loss = 0
                globalbuffers.dialog_optimizer.zero_grad()
            
                t1 = time.time()
                emb_gt = net.dialogmodel.embedding(Variable(masked_gt.view(bsz,-1))).view(bsz, -1, opt.seqlen, opt.emsize)
                output_dict, params = net.dialogmodel({'img': img, 'cap': targets['cap'], 'dialogblock': emb_gt}, sampler)

                t2 = time.time()
                
                ce_loss, kld_loss = criterion(extract_by_t(output_dict, Variable(t_ind)), extract_by_t(gt, Variable(t_ind)), params)
                numsamples = output_dict['tgt'].size(0)*1
                loss = (ce_loss + (klannealweight * kld_loss))/numsamples
                #p_x_given_zc = -1*ce_loss + p_z_given_c
                
                t3 = time.time()
             
                loss.backward() #must be true if there are parameters in vis_optimiser 
                
                t4 = time.time()
                
                globalbuffers.dialog_optimizer.step()
                
                t5 = time.time()
             
                ce_losses.update(ce_loss.data[0]/numsamples)
                kld_losses.update(kld_loss.data[0]/numsamples)
                losses.update(loss.data[0])
                #likelihoods.update(p_x_given_zc.data[0]/numsamples)
                grads.update(utils.sumParamGrads([net.vismodel, net.dialogmodel])/numsamples)

                if not opt.usepreds:
                    masked_gt.narrow(1,0,t+1).copy_(gt['tgt'].data.narrow(1,0,t+1))
                else:
    	            maxval, maxidx = torch.max(extract_by_t(output_dict, Variable(t_ind))['tgt'].squeeze(), dim=2)
    	            for b_zz in xrange(maxidx.size(0)):
    	                eos_pos = maxidx[b_zz].eq(dictionary.word2idx['EOS']).nonzero()
                        if eos_pos.size() and eos_pos.data[0][0]+1 < maxidx.size(1) :
     	                    eos_to_end = range(eos_pos.data[0][0]+1, maxidx.size(1))
    	                    #maxidx[b_zz][eos_to_end].data = torch.Tensor(1).fill_(dictionary.word2idx['PAD']).expand(len(eos_to_end)).type_as(gt['tgt'].data)
    	                    maxidx[b_zz][eos_to_end].data.fill_(dictionary.word2idx['PAD'])
			    del eos_to_end
                    if 'question' in opt.input_vars and 'answer' in opt.input_vars:
                        masked_gt.narrow(1,t,1).copy_(maxidx.unsqueeze(1).data) # predicted questions and answers
                    elif 'answer' in opt.input_vars:
                        masked_gt.narrow(1,t-1,1).copy(gt['tgt'].data.narrow(1,t-1,1)) #ground truth questions
                        masked_gt.narrow(1,t,1).copy(maxidx.unsqueeze(1).data) # predicted answers
		    del maxval, maxidx, eos_pos

		del emb_gt

             
        elif opt.blockmode == 'none':
            
            if 'item' in opt.input_vars: #  encode item at each t
                t_inds = torch.arange(0,opt.exchangesperimage*2).type_as(gt['tgt'].data).long().view(-1,1)
            else: # encode only answers  
                t_inds = torch.arange(1,opt.exchangesperimage*2,2).type_as(gt['tgt'].data).long().view(-1,1)
            
            all_inds = torch.arange(0,opt.exchangesperimage*2).type_as(gt['tgt'].data).long().view(-1,1)
            mask = torch.zeros(bsz, opt.exchangesperimage*2,opt.seqlen,opt.emsize).type_as(gt['tgt'].data).float().fill_(0)
            
            predicted_dialogue = gt['tgt'].data.clone()
            
            for t, t_ind in enumerate(t_inds):
            
                loss = 0
                globalbuffers.dialog_optimizer.zero_grad()
            
                t1 = time.time()
                emb_history = net.dialogmodel.embedding(Variable(predicted_dialogue).view(bsz,-1)).view(bsz, -1, opt.seqlen, opt.emsize)
                if 'item' in opt.input_vars:
                    emb_item = net.dialogmodel.embedding(gt['tgt'].index_select(1,Variable(t_ind)).view(bsz,-1)).view(bsz, -1, opt.seqlen, opt.emsize)
                    output_dict, params = net.dialogmodel({'img': img, 'cap': targets['cap'], 'item': emb_item, 'history': emb_history * Variable(mask)}, sampler)
                else:
                    emb_question = net.dialogmodel.embedding(gt['tgt'].index_select(1,Variable(t_ind-1)).view(bsz,-1)).view(bsz, -1, opt.seqlen, opt.emsize)
                    emb_answer = net.dialogmodel.embedding(gt['tgt'].index_select(1,Variable(t_ind)).view(bsz,-1)).view(bsz, -1, opt.seqlen, opt.emsize)
                    output_dict, params = net.dialogmodel({'img': img, 'cap': targets['cap'], 'question': emb_question, 'history': emb_history * Variable(mask), 'answer': emb_answer}, sampler)
            
                t2 = time.time()
             
                ce_loss, kld_loss = criterion(output_dict, extract_by_t(gt,Variable(t_ind)), params)
                numsamples = output_dict['tgt'].size(0)*output_dict['tgt'].size(1)
                loss = (ce_loss + (klannealweight * kld_loss))/numsamples
                #p_x_given_zc = -1*ce_loss + p_z_given_c
                t3 = time.time()
             
                if opt.verbose and (i % opt.log_interval*50) == 0:
                    print_dict = extract_by_t(gt, tgt_inds)
                    print_dict['dec'] = output_dict['tgt']
                    foutput = utils.formatoutput(print_dict, dictionary, opt.seqlen)
                    for (k,v) in foutput.iteritems():
                        printidx = 0 
                        print k + ": " + v[printidx]
            
                loss.backward(retain_graph=True) #must be true if there are parameters in vis_optimiser 
                mask.index_fill_(1, all_inds[all_inds <= t_ind].squeeze(), 1)
                
                if opt.usepreds:
    	            maxval, maxidx = torch.max(output_dict['tgt'].squeeze(), dim=2)
    	            for b_zz in xrange(maxidx.size(0)):
    	                eos_pos = maxidx[b_zz].eq(dictionary.word2idx['EOS']).nonzero()
                        if eos_pos.size() and eos_pos.data[0][0]+1 < maxidx.size(1) :
     	                    eos_to_end = range(eos_pos.data[0][0]+1, maxidx.size(1))
    	                    #maxidx[b_zz][eos_to_end].data = torch.Tensor(1).fill_(dictionary.word2idx['PAD']).expand(len(eos_to_end)).type_as(gt['tgt'].data)
    	                    maxidx[b_zz][eos_to_end].data.fill_(dictionary.word2idx['PAD'])
			    del eos_to_end
                    if 'item' in opt.input_vars:
                        predicted_dialogue.narrow(1,t,1).copy_(maxidx.unsqueeze(1).data) # predicted questions and answers
                    elif 'answer' in opt.input_vars: 
                        predicted_dialogue.narrow(1,t-1,1).copy(gt['tgt'].data.narrow(1,t-1,1)) #ground truth questions
                        predicted_dialogue.narrow(1,t,1).copy(maxidx.unsqueeze(1).data) # predicted answers
		    del maxval, maxidx, eos_pos


                #torch.nn.utils.clip_grad_norm(net.dialogmodel.parameters(), opt.clip)

                t4 = time.time()
                globalbuffers.dialog_optimizer.step()
                t5 = time.time()
             
                ce_losses.update(ce_loss.data[0]/numsamples)
                kld_losses.update(kld_loss.data[0]/numsamples)
                losses.update(loss.data[0])
                #likelihoods.update(p_x_given_zc.data[0]/numsamples)
                grads.update(utils.sumParamGrads([net.vismodel, net.dialogmodel])/numsamples)
                #print ('t == encode/decode: ' + str(t2-t1) + '; compute loss: ' + str(t3-t2) + '; back: ' + str(t4-t3) + '; step: ' + str(t5-t4))

                del emb_history
        

        if globalbuffers.vis_optimizer: #if no learnable parameters in vismodel
            torch.nn.utils.clip_grad_norm(net.vismodel.parameters(), opt.clip)
            globalbuffers.vis_optimizer.step()
        
        del gt
        del context
        del params
        del output_dict
        del img
        del targets
        del ce_loss
        del kld_loss
        #del p_z_given_c
        del loss
        #dot = plotting.make_dot(loss)
        #dot.view() 
        #print dot.source
        #raw_input()
        #clip_gradient(vqamodel, 0)
        

        #clipped_lr = opt.lr * clip_gradient(vqamodel, opt.clip)
        #for p in vqamodel.parameters():
        #    p.data.add_(-clipped_lr, p.grad.data)
        
        batch_time.update(time.time() - end)
         
        if i % opt.log_interval == 0:
            log.info('Epoch: [{0}][{1}/{2}]\t'
                  'GParams: {grad.val: 9.3f} ({grad.avg: 9.3f})\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'CEL: {cel.val:8.4f} ({cel.avg:8.4f})\t'
                  'KLD: {kld.val:8.4f} ({kld.avg:8.4f})\t'
                  'Loss: {loss.val:8.4f} ({loss.avg:8.4f})\t'
                  'LLHood: {lhood.val:8.4f} ({lhood.avg:8.4f})'.format( ep, i+1, len(data_loader), grad=grads, batch_time=batch_time, cel=ce_losses, kld=kld_losses, loss=losses,
                     lhood=likelihoods))

    if ep>=1:# % 4 == 0 or ep == 1:
        with open(os.path.join(resultsdir, 'models', logfilename + '_epoch' + str(ep) + '.pt'), 'wb') as f:
            net.save(f)
        if globalbuffers.vis_optimizer:
            with open(os.path.join(resultsdir, 'models', logfilename + '_visoptimizer.pt'), 'w') as vo:
                torch.save(globalbuffers.vis_optimizer.state_dict(), vo)
        with open(os.path.join(resultsdir, 'models', logfilename + '_dialogoptimizer.pt'), 'w') as do:
            torch.save(globalbuffers.dialog_optimizer.state_dict(), do)

    torch.save(ce_losses, os.path.join(resultsdir, 'logs', logfilename + '_epoch' + str(ep) + '_train_celoss.meter'))
    torch.save(kld_losses, os.path.join(resultsdir, 'logs', logfilename + '_epoch' + str(ep) + '_train_kldloss.meter'))
    torch.save(losses, os.path.join(resultsdir, 'logs', logfilename + '_epoch' + str(ep) + '_train_loss.meter'))
    torch.save(likelihoods, os.path.join(resultsdir, 'logs', logfilename + '_epoch' + str(ep) + '_train_likelihood.meter'))
    
