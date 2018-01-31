import torch
import torch.nn as nn
from torch.autograd import Variable 

enc_archies = {}
dec_archies = {}
prior_archies = {}
fBase = 16

# standard convolution encoder
# 4x4 conv filter
# input 1: (64 sentence length, 256 word embedding size)
enc_archies['conv_4x4_512_(1,64,256)'] = nn.Sequential(
            # input size: 1 x 64 x 256
            nn.Conv2d(1, fBase, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            # size: (fBase) x 32 x 128
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 16 x 64
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 4) x 8 x 32
            nn.Conv2d(fBase * 4, fBase * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 8),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 16
            nn.Conv2d(fBase * 8, fBase * 16, (1,4), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(fBase * 16),
            nn.ReLU(True),
            # size: (fBase * 16) x 4 x 8
            nn.Conv2d(fBase * 16, fBase * 16, (1,4), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(fBase * 16),
            nn.ReLU(True)
            # size: (fBase * 16) x 4 x 4
        )

dec_archies['conv_4x4_512_(1,64,256)'] = nn.Sequential(
            # size: 512 x 1 x 1
            nn.ConvTranspose2d(512, fBase * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(fBase * 16),
            nn.ReLU(True),
            # size: (fBase * 16) x 4 x 4
            nn.ConvTranspose2d(fBase * 16, fBase * 16, (1,4), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(fBase * 16),
            nn.ReLU(True),
            # size: (fBase * 16) x 4 x 8
            nn.ConvTranspose2d(fBase * 16, fBase * 8, (1,4), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(fBase * 8),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 16
            nn.ConvTranspose2d(fBase * 8, fBase * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 4) x 8 x 32
            nn.ConvTranspose2d(fBase * 4, fBase * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 16 x 64
            nn.ConvTranspose2d(fBase * 2, fBase, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            # size: (fBase) x 32 x 128
            nn.ConvTranspose2d(fBase, 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(True)
            # size: 1 x 64 x 256
        )

# standard convolution encoder
# 4x4 conv filter
# input 1: (64 sentence length, 256 word embedding size)
# input 2: (64 sentence length, 40 part of speech embedding size)
enc_archies['conv_4x4_512_(1,64,256)_(pos,i,1,64,40)'] = nn.Sequential(
            # input size: 1 x 64 x 296
            nn.Conv2d(1, fBase, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            # size: (fBase) x 32 x 148
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 16 x 74
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 4) x 8 x 37
            nn.Conv2d(fBase * 4, fBase * 8, (4,3), 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 8),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 19
            nn.Conv2d(fBase * 8, fBase * 8, (1,3), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(fBase * 8),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 10
            nn.Conv2d(fBase * 8, fBase * 16, (1,4), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(fBase * 16),
            nn.ReLU(True),
            # size: (fBase * 16) x 4 x 5
            nn.Conv2d(fBase * 16, fBase * 16, (1,2), 1, 0, bias=False),
            nn.BatchNorm2d(fBase * 16),
            nn.ReLU(True)
            # size: (fBase * 16) x 4 x 4
        )

dec_archies['conv_4x4_512_(1,64,256)_(pos,i,1,64,40)'] = nn.Sequential(
            # size: 512 x 1 x 1
            nn.ConvTranspose2d(512, fBase * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(fBase * 16),
            nn.ReLU(True),
            # size: (fBase * 16) x 4 x 4
            nn.ConvTranspose2d(fBase * 16, fBase * 16, (1,2), 1, 0, bias=False),
            nn.BatchNorm2d(fBase * 16),
            nn.ReLU(True),
            # size: (fBase * 16) x 4 x 5
            nn.ConvTranspose2d(fBase * 16, fBase * 16, (1,4), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(fBase * 16),
            nn.ReLU(True),
            # size: (fBase * 16) x 4 x 10
            nn.ConvTranspose2d(fBase * 16, fBase * 8, (1,3), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(fBase * 8),
            nn.ReLU(True),
            # size: (fBase * 16) x 4 x 19
            nn.ConvTranspose2d(fBase * 8, fBase * 4, (4,3), 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 4) x 8 x 37
            nn.ConvTranspose2d(fBase * 4, fBase * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 16 x 74
            nn.ConvTranspose2d(fBase * 2, fBase, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            # size: (fBase) x 32 x 148
            nn.ConvTranspose2d(fBase, 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(True)
            # size: 1 x 64 x 296
        )

# standard convolution encoder
# 4x4 conv filter
# input 1: (2*64 sentence length, 256 word embedding size)
enc_archies['conv_4x4_512_(2,64,256)'] = nn.Sequential(
            # input size: 2 x 64 x 256
            nn.Conv2d(2, fBase, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            # size: (fBase) x 32 x 128
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 16 x 64
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 4) x 8 x 32
            nn.Conv2d(fBase * 4, fBase * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 8),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 16
            nn.Conv2d(fBase * 8, fBase * 16, (1,4), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(fBase * 16),
            nn.ReLU(True),
            # size: (fBase * 16) x 4 x 8
            nn.Conv2d(fBase * 16, fBase * 16, (1,4), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(fBase * 16),
            nn.ReLU(True)
            # size: (fBase * 16) x 4 x 4
        )

dec_archies['conv_4x4_512_(2,64,256)'] = nn.Sequential(
            # size: 512 x 1 x 1
            nn.ConvTranspose2d(512, fBase * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(fBase * 16),
            nn.ReLU(True),
            # size: (fBase * 16) x 4 x 4
            nn.ConvTranspose2d(fBase * 16, fBase * 16, (1,4), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(fBase * 16),
            nn.ReLU(True),
            # size: (fBase * 16) x 4 x 8
            nn.ConvTranspose2d(fBase * 16, fBase * 8, (1,4), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(fBase * 8),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 16
            nn.ConvTranspose2d(fBase * 8, fBase * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 4) x 8 x 32
            nn.ConvTranspose2d(fBase * 4, fBase * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 16 x 64
            nn.ConvTranspose2d(fBase * 2, fBase, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            # size: (fBase) x 32 x 128
            nn.ConvTranspose2d(fBase, 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(True)
            # size: 2 x 64 x 256
        )
# full dialog block convolution encoder
# 4x4 conv filter
# input 1: (20 exchanges, 64 sentence length, 256 word embedding size)
enc_archies['conv_4x4_512_(20,64,256)'] = nn.Sequential(
            # input size: 20 x 64 x 256
            nn.Conv2d(20, fBase*2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(fBase*2),
            nn.ReLU(True),
            # size: (fBase) x 64 x 256
            nn.Conv2d(fBase*2, fBase*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase*2),
            nn.ReLU(True),
            # size: (fBase * 2) x 32 x 128
            nn.Conv2d(fBase*2, fBase*4, (1,4), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(fBase*4),
            nn.ReLU(True),
            # size: (fBase * 4) x 32 x 64
            nn.Conv2d(fBase*4, fBase*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase*4),
            nn.ReLU(True),
            # size: (fBase * 4) x 16 x 32
            nn.Conv2d(fBase*4, fBase*8, (1,4), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(fBase*8),
            nn.ReLU(True),
            # size: (fBase*8) x 16 x 16
            nn.Conv2d(fBase*8, fBase*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase*8),
            nn.ReLU(True),
            # size: (fBase*8) x 8 x 8
            nn.Conv2d(fBase*8, fBase*16, 1, 1, 0, bias=False),
            nn.BatchNorm2d(fBase*16),
            nn.ReLU(True),
            # size: (fBase*16) x 8 x 8
            nn.Conv2d(fBase*16, fBase*16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase*16),
            nn.ReLU(True)
            # size: (fBase*16) x 4 x 4
        )

dec_archies['conv_4x4_512_(20,64,256)'] = nn.Sequential(
            # size: 512 x 1 x 1
            nn.ConvTranspose2d(512, fBase * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(fBase*16),
            nn.ReLU(True),
            # size: (fBase*16) x 4 x 4
            nn.ConvTranspose2d(fBase*16, fBase*16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase*16),
            nn.ReLU(True),
            # size: (fBase*16) x 8 x 8
            nn.ConvTranspose2d(fBase*16, fBase*8, 1, 1, 0, bias=False),
            nn.BatchNorm2d(fBase*8),
            nn.ReLU(True),
            # size: (fBase*8) x 8 x 8
            nn.ConvTranspose2d(fBase*8, fBase*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase*8),
            nn.ReLU(True),
            # size: (fBase*8) x 16 x 16
            nn.ConvTranspose2d(fBase*8, fBase*4, (1,4), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(fBase*4),
            nn.ReLU(True),
            # size: (fBase*4) x 16 x 32
            nn.ConvTranspose2d(fBase*4, fBase*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase*4),
            nn.ReLU(True),
            # size: (fBase*4) x 32 x 64
            nn.ConvTranspose2d(fBase*4, fBase*2, (1,4), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(fBase*2),
            nn.ReLU(True),
            # size: (fBase*2) x 32 x 128
            nn.ConvTranspose2d(fBase*2, fBase*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase*2),
            nn.ReLU(True),
            # size: (fBase*2) x 64 x 256
            nn.ConvTranspose2d(fBase*2, 20, 1, 1, 0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU(True)
            # size: 20 x 64 x 256
        )

# full dialog block convolution encoder with image conditioning
# 4x4 conv filter
# input 1: (20 exchanges, 64 sentence length, 256 word embedding size)
# condition 1: (512, 1, 1) resnet image feature
enc_archies['conv_4x4_512_(20,64,256)_(img,c,512)'] = nn.ModuleList(
        [nn.Sequential(
            # input size: 20 x 64 x 256
            nn.Conv2d(20, fBase*2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(fBase*2),
            nn.ReLU(True),
            # size: (fBase) x 64 x 256
            nn.Conv2d(fBase*2, fBase*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase*2),
            nn.ReLU(True),
            # size: (fBase * 2) x 32 x 128
            nn.Conv2d(fBase*2, fBase*4, (1,4), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(fBase*4),
            nn.ReLU(True),
            # size: (fBase * 4) x 32 x 64
            nn.Conv2d(fBase*4, fBase*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase*4),
            nn.ReLU(True)
            # size: (fBase * 4) x 16 x 32
            ),
        nn.Sequential(
            nn.Conv2d(fBase*4+1, fBase*8, (1,4), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(fBase*8),
            nn.ReLU(True),
            # size: (fBase*8) x 16 x 16
            nn.Conv2d(fBase*8, fBase*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase*8),
            nn.ReLU(True),
            # size: (fBase*8) x 8 x 8
            nn.Conv2d(fBase*8, fBase*16, 1, 1, 0, bias=False),
            nn.BatchNorm2d(fBase*16),
            nn.ReLU(True),
            # size: (fBase*16) x 8 x 8
            nn.Conv2d(fBase*16, fBase*16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase*16),
            nn.ReLU(True)
            # size: (fBase*16) x 4 x 4
        )])

enc_archies['history_(20,64,256)'] = nn.Sequential(
            # input size: 20 x 64 x 256
            nn.Conv2d(20, fBase*2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(fBase*2),
            nn.ReLU(True),
            # size: (fBase*2) x 64 x 256
            nn.Conv2d(fBase*2, fBase*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase*2),
            nn.ReLU(True),
            # size: (fBase*2) x 32 x 128
            nn.Conv2d(fBase*2, fBase*2, (1,4), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(fBase*2),
            nn.ReLU(True),
            # size: (fBase*2) x 32 x 64
            nn.Conv2d(fBase*2, fBase*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase*4),
            nn.ReLU(True),
            # size: (fBase*4) x 16 x 32
            nn.Conv2d(fBase*4, fBase*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase*4),
            nn.ReLU(True),
            # size: (fBase*4) x 8 x 16
            nn.Conv2d(fBase*4, fBase*4, (1,4), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(fBase*4),
            nn.ReLU(True)
            # size: (fBase*4) x 8 x 8
            )

enc_archies['dialogblock_(20,64,256)'] = nn.Sequential(
            # input size: 20 x 64 x 256
            nn.Conv2d(20, fBase*2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(fBase*2),
            nn.ReLU(True),
            # size: (fBase*2) x 64 x 256
            nn.Conv2d(fBase*2, fBase*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase*2),
            nn.ReLU(True),
            # size: (fBase*2) x 32 x 128
            nn.Conv2d(fBase*2, fBase*2, (1,4), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(fBase*2),
            nn.ReLU(True),
            # size: (fBase*2) x 32 x 64
            nn.Conv2d(fBase*2, fBase*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase*4),
            nn.ReLU(True),
            # size: (fBase*4) x 16 x 32
            nn.Conv2d(fBase*4, fBase*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase*4),
            nn.ReLU(True),
            # size: (fBase*4) x 8 x 16
            nn.Conv2d(fBase*4, fBase*4, (1,4), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(fBase*4),
            nn.ReLU(True)
            # size: (fBase*4) x 8 x 8
            )

enc_archies['item_(1,64,256)'] = nn.Sequential(
            # input size: 1 x 64 x 256
            nn.Conv2d(1, fBase, (4,6), (2,4), 1, bias=False),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            # size: (fBase) x 32 x 64
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 16 x 32
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 4) x 8 x 16
            nn.Conv2d(fBase * 4, fBase * 4, (1,4), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True)
            # size: (fBase * 4) x 8 x 8
            )

enc_archies['item_(1,64,300)'] = nn.Sequential(
            # input size: 1 x 64 x 300
            nn.Conv2d(1, fBase, (4,7), (2,5), 1, bias=False),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            # size: (fBase) x 32 x 60
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 16 x 30
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 4) x 8 x 15
            nn.Conv2d(fBase * 4, fBase * 4, (1,3), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True)
            # size: (fBase * 4) x 8 x 8
            )
            
enc_archies['joint_(64,8,8)'] = nn.Sequential(
            # size: 64 x 8 x 8
            nn.Conv2d(fBase * 4, fBase * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 8),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 4
            nn.Conv2d(fBase * 8, fBase * 16, 1, 1, 0, bias=False),
            nn.BatchNorm2d(fBase * 16),
            nn.ReLU(True)
            # size: (fBase*16) x 4 x 4
        )

enc_archies['answer_(1,64,256)'] = nn.Sequential(
            # input size: 1 x 64 x 256
            nn.Conv2d(1, fBase, (4,6), (2,4), 1, bias=False),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            # size: (fBase) x 32 x 64
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 16 x 32
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 4) x 8 x 16
            nn.Conv2d(fBase * 4, fBase * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 8),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 8
            nn.Conv2d(fBase * 8, fBase * 16, (1,4), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(fBase * 16),
            nn.ReLU(True)
            # size: (fBase * 16) x 4 x 4
        )

dec_archies['latent_(512,1,1)'] = nn.Sequential(
            # size: 512 x 1 x 1
            nn.ConvTranspose2d(512, fBase*16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(fBase*16),
            nn.ReLU(True),
            # size: (fBase * 16) x 4 x 4
            nn.ConvTranspose2d(fBase*16, fBase*8, 1, 1, 0, bias=False),
            nn.BatchNorm2d(fBase*8),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 4
            nn.ConvTranspose2d(fBase*8, fBase*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase*4),
            nn.ReLU(True)
            # size: 64 x 8 x 8
            )

dec_archies['latent_and_condition_(64,8,8)'] = nn.Sequential(
            # size: 64 x 8 x 8
            nn.ConvTranspose2d(fBase * 4, fBase * 4, (1,4), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 4) x 8 x 16
            nn.ConvTranspose2d(fBase * 4, fBase * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 16 x 32
            nn.ConvTranspose2d(fBase * 2, fBase, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            # size: (fBase) x 32 x 64
            nn.ConvTranspose2d(fBase, 1, (4,6), (2,4), 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(True)
            # size: 1 x 64 x 256
        )

dec_archies['latent_and_condition_(64,8,8)_dialogblock'] = nn.Sequential(
            # size: 64 x 8 x 8
            nn.ConvTranspose2d(fBase*4, fBase*4, (1,4), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(fBase*4),
            nn.ReLU(True), 
            # size: (fbase*4) x 8 x 16
            nn.ConvTranspose2d(fBase*4, fBase*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase*4),
            nn.ReLU(True), 
            # size: (fbase*4) x 16 x 32 
            nn.ConvTranspose2d(fBase*4, fBase*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase*2),
            nn.ReLU(True), 
            # size: (fbase*2) x 32 x 64 
            nn.ConvTranspose2d(fBase*2, fBase*2, (1,4), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(fBase*2),
            nn.ReLU(True),
            # size: (fbase*2) x 32 x 128
            nn.ConvTranspose2d(fBase*2, fBase*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase*2),
            nn.ReLU(True),
            # size: (fBase*2) x 64 x 256
            nn.ConvTranspose2d(fBase*2, 20, 1, 1, 0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU(True)
            # size: 20 x 64 x 256
        )


prior_archies['question_(1,64,256)'] = nn.Sequential(
            # input size: 1 x 64 x 256
            nn.Conv2d(1, fBase, (4,6), (2,4), 1, bias=False),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            # size: (fBase) x 32 x 64
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 16 x 32
            nn.Conv2d(fBase * 2, fBase * 4, (4,6), (2,4), 1, bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 4) x 8 x 8
            nn.Conv2d(fBase * 4, fBase * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 8),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 4
            nn.Conv2d(fBase * 8, fBase * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 16),
            nn.ReLU(True),
            # size: (fBase * 16) x 2 x 2
            nn.Conv2d(fBase * 16, fBase * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 32),
            nn.ReLU(True)
            # size: (fBase * 32) x 1 x 1
        )

# 1QA dialog block convolution encoder with image conditioning
# 4x4 conv filter
# input 1: (2 exchanges, 64 sentence length, 256 word embedding size)
# condition 1: (512, 1, 1) resnet image feature
enc_archies['conv_4x4_512_(2,64,256)_(img,c,512)'] = nn.ModuleList(
        [nn.Sequential(
            # input size: 20 x 64 x 256
            nn.Conv2d(2, fBase*2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(fBase*2),
            nn.ReLU(True),
            # size: (fBase) x 64 x 256
            nn.Conv2d(fBase*2, fBase*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase*2),
            nn.ReLU(True),
            # size: (fBase * 2) x 32 x 128
            nn.Conv2d(fBase*2, fBase*4, (1,4), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(fBase*4),
            nn.ReLU(True),
            # size: (fBase * 4) x 32 x 64
            nn.Conv2d(fBase*4, fBase*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase*4),
            nn.ReLU(True)
            # size: (fBase * 4) x 16 x 32
            ),
        nn.Sequential(
            nn.Conv2d(fBase*4+1, fBase*8, (1,4), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(fBase*8),
            nn.ReLU(True),
            # size: (fBase*8) x 16 x 16
            nn.Conv2d(fBase*8, fBase*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase*8),
            nn.ReLU(True),
            # size: (fBase*8) x 8 x 8
            nn.Conv2d(fBase*8, fBase*16, 1, 1, 0, bias=False),
            nn.BatchNorm2d(fBase*16),
            nn.ReLU(True),
            # size: (fBase*16) x 8 x 8
            nn.Conv2d(fBase*16, fBase*16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase*16),
            nn.ReLU(True)
            # size: (fBase*16) x 4 x 4
        )])

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        #self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + (mask_type == 'B'):] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

dec_archies['autoreg8_1_64_9710'] = nn.Sequential(
    MaskedConv2d('A', 1,  64, 7, 1, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
    MaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
    MaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
    MaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
    MaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
    MaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
    MaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
    MaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(True))

dec_archies['autoreg5_256_256_9710'] = nn.Sequential(
    MaskedConv2d('A', 256,  256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True))

dec_archies['autoreg8_256_256_9710'] = nn.Sequential(
    MaskedConv2d('A', 256,  256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True))

dec_archies['autoreg5_noA_256_256_9710'] = nn.Sequential(
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True))

dec_archies['autoreg5_2x_noA_256_256_9710'] = nn.Sequential(
    MaskedConv2d('B', 256, 256, (15,1), 1, (7,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (15,1), 1, (7,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (15,1), 1, (7,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (15,1), 1, (7,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (15,1), 1, (7,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True))

dec_archies['autoreg8_noA_256_256_9710'] = nn.Sequential(
    MaskedConv2d('B', 256,  256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True))

dec_archies['autoreg10_noA_256_256_9710'] = nn.Sequential(
    MaskedConv2d('B', 256,  256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (7,1), 1, (3,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True))

dec_archies['autoreg8_2x_noA_256_256_9710'] = nn.Sequential(
    MaskedConv2d('B', 256,  256, (15,1), 1, (7,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (15,1), 1, (7,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (15,1), 1, (7,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (15,1), 1, (7,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (15,1), 1, (7,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (15,1), 1, (7,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (15,1), 1, (7,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
    MaskedConv2d('B', 256, 256, (15,1), 1, (7,0), bias=False), nn.BatchNorm2d(256), nn.ReLU(True))
