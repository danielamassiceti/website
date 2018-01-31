import inflect, re, torch, inflect
from sklearn.metrics import pairwise
from nltk import word_tokenize
import numpy as np

from torch.autograd import Variable
from torch.nn import functional as F

verbose = False
EOS_ID = 1
nc = inflect.engine()

def preprocess(sentence, seqlen, dictionary):
    
    toks = []
    tokens = standardiselength(word_tokenize(standardizechars(sentence, nc)), seqlen)
    for t in tokens:
        toks.append(dictionary.word2idx.get(t, dictionary.UNK_ID))

    return toks

# convert digits to words
def digitreplacer(digit, numconverter):
    return numconverter.number_to_words((digit)).replace("-"," ")

# gets token list to be of length opt.seqlen using EOS, GO and PAD tokens
# answers: first token is always GO, last token is always EOS
# if questions/answers are shorter than opt.seqlen, then pad with PAD
# if questions/answers are longer than opt.seqlen, then chop-chop!
def standardiselength(tokenlist, seqlen):
    padtoken = 'PAD'.encode('utf-8')
    eostoken = 'EOS'.encode('utf-8')
    paddedtokens = [padtoken] * seqlen
    for i,t in enumerate(tokenlist):
        if i < seqlen:
    	    paddedtokens[i] = t
	else:
	    break

    npad = len(tokenlist)-seqlen
    if npad < 0: #shorter
        paddedtokens[len(tokenlist)] = eostoken
    else:
    	paddedtokens[seqlen-1] = eostoken
    
    return paddedtokens
        
def standardizechars(sentence, numconverter):

    # remove all apostrophes
    sentence = sentence.replace("'", "")
    sentence = sentence.replace("?", "")
        
    return re.sub(r'\d+', lambda x: digitreplacer(x.group(), numconverter), sentence).lower()
     
def computelikelihood(probs, gt):
    
    # compute probabilities for each answer, based on its tokens
    predvec = probs.view(-1, probs.size(3)) 
    inds = gt.view(-1, 1)

    logprobs = torch.log(predvec.gather(1, inds)).view_as(gt)
    logprobs = logprobs.sum(dim=2)
    return logprobs

def caption_question_sim(dec_qs, cap, dictionary):
    
    # cap: bsz x 1 x seqlen
    # dec_answer: bsz x N x seqlen x vocabsize
    
    cap_wordvecs = wordvector_avg(cap.data, dictionary)
    sims = F.cosine_similarity( wordvector_avg(dec_qs.data, dictionary), cap_wordvecs.expand(cap_wordvecs.size(0), dec_qs.size(1), cap_wordvecs.size(2)), dim=2)

    return sims.mean()

def rankcandidates(dec_answer, candidates, gtidxs, dictionary):
    
    # dec_answer is bsz x N x seqlen x vocabsize
    # candidates is bsz x N x 100 x seqlen 
    # gtidx is bsz x N 
    # N is numexchanges
    
    cand_wordvecs = wordvector_avg(candidates.data, dictionary)
 
    sims = F.cosine_similarity( wordvector_avg(dec_answer.data, dictionary).unsqueeze(2).expand(cand_wordvecs.size(0), cand_wordvecs.size(1), cand_wordvecs.size(2), cand_wordvecs.size(3)), cand_wordvecs, dim=3)
    if sims.dim() == 2:
        sims = sims.unsqueeze(1)

    # sims: bsz x N x 100
    sorted_sims, indices = torch.sort(sims, dim=2, descending=True)
    ranks = sorted_sims.new(sorted_sims.size()).fill_(0).scatter_(2, indices, sorted_sims.new(torch.arange(1,101).type_as(sorted_sims)).unsqueeze(0).unsqueeze(1).expand_as(sorted_sims))

    gt_ranks = ranks.gather(2, gtidxs.data.contiguous().view(gtidxs.size(0), gtidxs.size(1),1))
    mr = gt_ranks.mean()
    k = [1, 5, 10]
    recall_at_k = [0 for j in xrange(len(k))]
    for i,kval in enumerate(k):
        recall_at_k[i] = torch.le(gt_ranks,kval).sum()/float(gt_ranks.size(0)*gt_ranks.size(1))*100
    mrr = (1/gt_ranks).sum()/float(gt_ranks.size(0)*gt_ranks.size(1))
    
    return mr, recall_at_k, mrr 
            
def rankandeval(embedding, rnnencoder, candidates, predans, gtidx, bsz):
    b = 0
    pred_hidden = rnnencoder.init_hidden(bsz)
    embans = embedding(predans.view(bsz,-1))
    for w in xrange(predans.size(0)):
        pred_hidden = rnnencoder(embans[b][w].view(1,-1), pred_hidden)
        if predans[w].data.eq(EOS_ID).any():
            break

    cand_hiddens = []
    for o in candidates:
        cand_hidden = rnnencoder.init_hidden(bsz)
        embans = embedding(o.view(bsz,-1))
        for w in xrange(o.size(0)):
            cand_hidden = rnnencoder(embans[b][w].view(1,-1), cand_hidden)
            if o[w].data.eq(EOS_ID).any():
                break
        cand_hiddens.append(cand_hidden[0])

    sim = []
    for r in xrange(candidates.size(0)):
        sim.append(pairwise.cosine_similarity(pred_hidden[0].data.cpu().numpy(), cand_hiddens[r].data.cpu().numpy())[0][0])

    # get MRR and mean rank
    # first element of ranks gives rank of predicted answer
    npsim = np.array(sim)
    temp = npsim.argsort()[::-1] #descending order (i.e. rank 0 is highest similarity value)
    ranks = np.empty(len(npsim), int)
    ranks[temp] = np.arange(1, len(npsim)+1) #(i.e. rank from 1 to 100)
    #for i in xrange(len(npsim)):
    #    print (str(ranks[i]) + ": " + str(npsim[i]))

    k = [1, 5, 10]
    recall_at_k = [0] * len(k)

    
    for i, kval in enumerate(k):
        if ranks[gtidx.data[0]] <= kval: recall_at_k[i] = 1

    return ranks[gtidx.data[0]], recall_at_k, 1/float(ranks[gtidx.data[0]])

def visualize_regions(image, regions):
    response = requests.get(image.url)
    img = PIL_Image.open(StringIO(response.content))
    plt.imshow(img)
    ax = plt.gca()
    for region in regions:
        ax.add_patch(Rectangle((region.x, region.y),
                               region.width,
                               region.height,
                               fill=False,
                               edgecolor='red',
                               linewidth=3))
        ax.text(region.x, region.y, region.phrase, style='italic', bbox={'facecolor':'white', 'alpha':0.7, 'pad':10})
    fig = plt.gcf()
    plt.tick_params(labelbottom='off', labelleft='off')
    plt.show()
    
def getsequencemask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand

def rankscores(scores, gtidx):
    gt_score = scores.gather(0, gtidx.view(1,-1))
    ranks = scores.lt(gt_score.expand_as(scores))
    ranks = ranks.sum(0) + 1
    return ranks.float() # in [1,100]
 
def get_w2v(idxbatch, dictionary):

    return dictionary.word_embs.index_select(0, idxbatch.view(-1)).view(idxbatch.size(0), idxbatch.size(1), idxbatch.size(2), -1)


def getidx_and_pad(sentencebatch):

    _, maxidx = sentencebatch.data.max(dim=3)
    maxidx = maxidx.view(maxidx.size(0)*maxidx.size(1), maxidx.size(2))
    eos_idxs = maxidx.eq(1).nonzero()
    for b in eos_idxs:
        if b[1]<maxidx.size(1)-1:
            maxidx[b[0], b[1]+1:maxidx.size(1)].fill_(0)
    return Variable(maxidx.view(sentencebatch.size(0), sentencebatch.size(1), sentencebatch.size(2)), volatile=sentencebatch.volatile)

def wordvector_avg(sentencebatch, dictionary, evaluation=False):

    vocabsize = len(dictionary)

    sentencebatch = sentencebatch.cpu()
   
    if sentencebatch.dim() == 4:
        if sentencebatch.size(3) == vocabsize:
            maxval, maxidx = sentencebatch.max(dim=3)
            wordvecs = dictionary.word_embs.index_select(0, maxidx.view(-1)).view(maxidx.size(0)*maxidx.size(1), maxidx.size(2), -1)
            eos_idxs = maxidx.view(-1, maxidx.size(2)).eq(1).nonzero()
            for b in eos_idxs:
                if b[1]<maxidx.size(2)-1:
                    wordvecs[b[0], b[1]+1:maxidx.size(2)].fill_(0)
            wordvecs = wordvecs.view(maxidx.size(0), maxidx.size(1), maxidx.size(2), -1)

            lens = wordvecs.sum(3).ne(0).sum(2)
            return wordvecs.sum(2) / lens.unsqueeze(2).expand(wordvecs.size(0), wordvecs.size(1), wordvecs.size(3)).type_as(wordvecs) 
    
        else:
            wordvecs = dictionary.word_embs.index_select(0, sentencebatch.contiguous().view(-1)).contiguous().view(sentencebatch.size(0)*sentencebatch.size(1)*sentencebatch.size(2), sentencebatch.size(3), -1)
            eos_idxs = sentencebatch.contiguous().view(-1, sentencebatch.size(3)).eq(1).nonzero()
            for b in eos_idxs:
                if b[1]<sentencebatch.size(3)-1:
                    wordvecs[b[0], b[1]+1:sentencebatch.size(3)].fill_(0)
            wordvecs = wordvecs.view(sentencebatch.size(0), sentencebatch.size(1), sentencebatch.size(2), sentencebatch.size(3), -1)

            lens = wordvecs.sum(4).ne(0).sum(3)
            return wordvecs.sum(3) / lens.unsqueeze(3).expand(wordvecs.size(0), wordvecs.size(1), wordvecs.size(2), wordvecs.size(4)).type_as(wordvecs) 
    elif sentencebatch.dim() == 3:
        wordvecs = dictionary.word_embs.index_select(0, sentencebatch.contiguous().view(-1)).contiguous().view(sentencebatch.size(0)*sentencebatch.size(1), sentencebatch.size(2), -1)
        eos_idxs = sentencebatch.contiguous().view(-1, sentencebatch.size(2)).eq(1).nonzero()
        for b in eos_idxs:
            if b[1]<sentencebatch.size(2)-1:
                wordvecs[b[0],b[1]+1:sentencebatch.size(2)].fill_(0)
        wordvecs = wordvecs.view(sentencebatch.size(0), sentencebatch.size(1), sentencebatch.size(2), -1)
        lens = wordvecs.sum(3).ne(0).sum(2)
        return wordvecs.sum(2) / lens.unsqueeze(2).expand(wordvecs.size(0), wordvecs.size(1), wordvecs.size(3)).type_as(wordvecs)

def tensor2string(tensor, dictionary, seqlen, withpos=False):
    bsz = tensor.size(0)
    nexchanges = tensor.size(1)
    output = [ ["" for i in xrange(nexchanges)] for j in xrange(bsz)]
    for b in xrange(bsz):
        for e in xrange(nexchanges):
            for w in xrange(seqlen):
                if tensor.dim() > 3:
                    maxval, maxidx = torch.max(tensor.data[b][e][w], 0)
                    if maxval.ne(dictionary.PAD_ID).all(): #not PAD
                        output[b][e] += " " + dictionary.idx2word[maxidx[0]]
                    else:
                        output[b][e] += " " + dictionary.idx2word[0]
                else:
                    output[b][e] += " " + dictionary.idx2word[tensor.data[b][e][w]]
            output[b][e] = (output[b][e][0:output[b][e].find('EOS')]).encode('utf-8', 'ignore')
    return output

def formatoutput(input, dictionary, seqlen): 
    numitems = len(input)
    assert(numitems > 0)
    output = {}

    for k,v in input.iteritems():
        if isinstance(v, list):
            lstsz = len(v)
            output[k] = []
            for l in v:
                output[k].append(tensor2string(l, dictionary, seqlen))
        else:
            output[k] = tensor2string(v, dictionary, seqlen)
    
    return output
