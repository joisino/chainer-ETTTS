import sys

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F

class Conv(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize, dilate=1, causal=True, dropout=0.1):
        self.causal = causal
        self.dropout = dropout
        if causal:
            self.pad = (ksize-1) * dilate
        else:
            self.pad = (ksize-1) * dilate // 2
        super(Conv, self).__init__(
            conv=L.DilatedConvolution2D(
                in_channels,
                out_channels,
                (1, ksize),
                1,
                (0, self.pad),
                (1, dilate),
                initialW=chainer.initializers.HeNormal(),
            ),
        )

    def __call__(self, x):
        h = F.expand_dims(x, 2)
        h = self.conv(h)
        if self.causal and self.pad > 0:
            h = h[..., 0, :-self.pad]
        else:
            h = h[..., 0, :]
        h = F.dropout(h, self.dropout)
        return h

    
class Highway(chainer.Chain):
    def __init__(self, d, k, delta, causal=True):
        self.d = d
        super(Highway, self).__init__(
            conv=Conv(d, 2*d, k, delta, causal)
        )

    def __call__(self, x):
        h1 = self.conv(x)
        h2 = h1[:, :self.d, ...]
        h3 = h1[:, self.d:, ...]
        h4 = F.sigmoid(h2)
        res = h4 * h3 + (1-h4) * x
        return res

class TextEnc(chainer.Chain):
    def __init__(self, s, e, d):
        super(TextEnc, self).__init__(
            l1=L.EmbedID(s, e, initialW=chainer.initializers.HeNormal()),
            l2=Conv(e, 2*d, 1, 1, False),
            l3=Conv(2*d, 2*d, 1, 1, False),
            l4=Highway(2*d, 3, 1, False),
            l5=Highway(2*d, 3, 3, False),
            l6=Highway(2*d, 3, 9, False),
            l7=Highway(2*d, 3, 27, False),
            l8=Highway(2*d, 3, 1, False),
            l9=Highway(2*d, 3, 3, False),
            l10=Highway(2*d, 3, 9, False),
            l11=Highway(2*d, 3, 27, False),
            l12=Highway(2*d, 3, 1, False),
            l13=Highway(2*d, 3, 1, False),
            l14=Highway(2*d, 1, 1, False),
            l15=Highway(2*d, 1, 1, False),
        )

    def __call__(self, x):
        h = self.l1(x)
        h = F.transpose(h, (0, 2, 1))
        h = F.relu(self.l2(h))
        for i in range(3, 16):
            h = self['l%d'%i](h)
        return h


class AudioEnc(chainer.Chain):
    def __init__(self, d, f):
        super(AudioEnc, self).__init__(
            l1=Conv(f, d, 1, 1),
            l2=Conv(d, d, 1, 1),
            l3=Conv(d, d, 1, 1),
            l4=Highway(d, 3, 1),
            l5=Highway(d, 3, 3),
            l6=Highway(d, 3, 9),
            l7=Highway(d, 3, 27),
            l8=Highway(d, 3, 1),
            l9=Highway(d, 3, 3),
            l10=Highway(d, 3, 9),
            l11=Highway(d, 3, 27),
            l12=Highway(d, 3, 3),
            l13=Highway(d, 3, 3),
        )

    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        for i in range(3, 14):
            h = self['l%d'%i](h)
        return h


class AudioDec(chainer.Chain):
    def __init__(self, d, f):
        super(AudioDec, self).__init__(
            l1=Conv(2*d, d, 1, 1),
            l2=Highway(d, 3, 1),
            l3=Highway(d, 3, 3),
            l4=Highway(d, 3, 9),
            l5=Highway(d, 3, 27),
            l6=Highway(d, 3, 1),
            l7=Highway(d, 3, 1),
            l8=Conv(d, d, 1, 1, dropout=0),
            l9=Conv(d, d, 1, 1, dropout=0),
            l10=Conv(d, d, 1, 1, dropout=0),
            l11=Conv(d, f, 1, 1, dropout=0),
        )

    def __call__(self, x):
        h = x
        for i in range(1, 8):
            h = self['l%d'%i](h)
        h = F.relu(self.l8(h))
        h = F.relu(self.l9(h))
        h = F.relu(self.l10(h))
        h = self.l11(h)
        return h


class SynthesisNetwork(chainer.Chain):
    def __init__(self, s=32, e=128, d=256, f=80, g=0.2):
        self.d = d
        self.g = g
        super(SynthesisNetwork, self).__init__(
            text_enc=TextEnc(s, e, d),
            audio_enc=AudioEnc(d, f),
            audio_dec=AudioDec(d, f),
        )

    def gen(self, text, x):
        vk = self.text_enc(text)
        v = vk[:, :self.d, :]
        k = vk[:, self.d:, :]
        q = self.audio_enc(x)

        a = F.matmul(F.transpose(k, (0, 2, 1)), q)
        a = F.softmax(a / self.xp.sqrt(self.d))

        prva = -1
        for i in range(a.shape[2]):
            if (self.xp.argmax(a.data[0, :, i]) < prva - 1
                or prva + 3 < self.xp.argmax(a.data[0, :, i])):
                a = a.data
                a[0, :, i] = np.zeros(a.shape[1], dtype='f')
                pos = min(a.shape[1]-1, prva+1)
                a[0, pos, i] = 1
                a = chainer.Variable(a)
            prva = self.xp.argmax(a.data[0, :, i])
        
        r = F.matmul(v, a)
        rd = F.concat((r, q))

        y = self.audio_dec(rd)
        y = F.sigmoid(y)
        
        return y.data, a.data
        
    def __call__(self, text, x, t, textlens, xlens):
        batchsize = text.shape[0]
        
        vk = self.text_enc(text)
        
        v = vk[:, :self.d, :]
        k = vk[:, self.d:, :]
        q = self.audio_enc(x)

        a = F.matmul(F.transpose(k, (0, 2, 1)), q)
        a = F.softmax(a / self.xp.sqrt(self.d))
        r = F.matmul(v, a)
        rd = F.concat((r, q))

        y = self.audio_dec(rd)

        loss_bin = 0
        for i in range(batchsize):
            loss_bin += F.mean(F.bernoulli_nll(t[i, :, :xlens[i]], y[i, :, :xlens[i]], 'no'))
        loss_bin /= batchsize
        
        y = F.sigmoid(y)

        loss_l1 = 0
        for i in range(batchsize):
            loss_l1 += F.mean_absolute_error(t[i, :, :xlens[i]], y[i, :, :xlens[i]])
        loss_l1 /= batchsize
            
        loss_att = 0
        for i in range(batchsize):
            N = textlens[i]
            T = xlens[i]
            def w_fun(n, t):
                return 1 - np.exp(-((n/(N-1) - t/(T-1))**2) / (2 * self.g**2))
            w = np.fromfunction(w_fun, (a.shape[1], T), dtype='f')
            w = self.xp.array(w)
            loss_att += F.mean(w * a[i, :, :T])
        loss_att /= batchsize

        loss = loss_bin + loss_l1 + loss_att

        chainer.reporter.report(
            {
                'loss_bin': loss_bin,
                'loss_l1': loss_l1,
                'loss_att': loss_att,
            }
        )
        
        return loss, y, a
        
