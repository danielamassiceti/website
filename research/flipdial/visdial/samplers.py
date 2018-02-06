import torch, math
import torch.nn as nn
from numpy import pi
from torch.autograd import Variable

class GaussianSampler(nn.Module):
    """ Draw a sample from a Gaussian distribution. """
    def __init__(self):
        super(GaussianSampler, self).__init__()

    def kld(self, latent_params):
        if 'mu_q' in latent_params and 'mu_p' in latent_params:
            """ Compute D_kl( N(mu_q, exp(logvar_q)) || N(mu_p, exp(logvar_p) ). """
            mu_q = latent_params['mu_q']
            logvar_q = latent_params['logvar_q']
            mu_p = latent_params['mu_p']
            logvar_p = latent_params['logvar_p']
            _kl = (mu_q - mu_p).pow(2).add_(logvar_q.exp().add(1e-6)).div(-1*(logvar_p.exp().add(1e-6))).add_(1).add_(logvar_q).add_(-1*logvar_p)
            # _kl = (mu_q - mu_p).pow(2).add_(logvar_q.exp()).div(-1*(logvar_p.exp())).add_(1).add_(logvar_q).add_(-1*logvar_p)
            return torch.sum(_kl).mul_(-0.5)
        elif 'mu_q' in latent_params:
            """ Compute D_kl( N(mu_q, exp(logvar_q)) || N(0,1) ). """
            mu_q = latent_params['mu_q']
            logvar_q = latent_params['logvar_q']
            _kl = mu_q.pow(2).add_(logvar_q.exp().add(1e-6)).mul_(-1).add_(1).add_(logvar_q)
            #_kl = mu_q.pow(2).add_(logvar_q.exp()).mul_(-1).add_(1).add_(logvar_q)
            return torch.sum(_kl).mul_(-0.5)
        elif 'mu_p' in latent_params:
            """ Compute D_kl( N(mu_p, exp(logvar_p)) || N(0,1) ). """
            mu_p = latent_params['mu_p']
            logvar_p = latent_params['logvar_p']
            _kl = mu_p.pow(2).add_(logvar_p.exp().add(1e-6)).mul_(-1).add_(1).add_(logvar_p)
            #_kl = mu_q.pow(2).add_(logvar_q.exp()).mul_(-1).add_(1).add_(logvar_q)
            return torch.sum(_kl).mul_(-0.5)

    def logpdf(self, z, mu, logvar, leavebatch=True):
	if z.dim() > 4: #nsamples > 1
	    mu = mu.unsqueeze(0).expand(z.size(0), -1, -1, -1, -1)
	    logvar = logvar.unsqueeze(0).expand(z.size(0), -1, -1, -1, -1)
        lpdf = (z - mu).pow(2).div(logvar.exp())
        if leavebatch:
            if z.dim() > 4:
                return lpdf.add_(logvar).add_(math.log(2*math.pi)).mul_(-0.5).sum(2)
            else:
                return lpdf.add_(logvar).add_(math.log(2*math.pi)).mul_(-0.5).sum(1)
        else:
            return lpdf.add_(logvar).add_(math.log(2*math.pi)).mul_(-0.5).sum()

    def forward(self, mu, logvar, evaluation):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        eps = Variable(eps, volatile=evaluation)
        return eps.mul(std).add_(mu)
