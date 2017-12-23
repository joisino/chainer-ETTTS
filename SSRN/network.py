import sys

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F

class Conv(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize, dilate=1, causal=True):
        self.causal = causal
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
                initialW=chainer.initializers.HeNormal()
            ),
        )

    def __call__(self, x):
        h = F.expand_dims(x, 2)
        h = self.conv(h)
        if self.causal and self.pad > 0:
            h = h[..., 0, :-self.pad]
        else:
            h = h[..., 0, :]
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

class SSRN(chainer.Chain):
    def __init__(self, f, c, fd):
        super(SSRN, self).__init__(
            l1=Conv(f, c, 1, 1, False),
            l2=Highway(c, 3, 1, False),
            l3=Highway(c, 3, 3, False),
            l4=L.DeconvolutionND(1, c, c, 2, 2, initialW=chainer.initializers.HeNormal()),
            l5=Highway(c, 3, 1, False),
            l6=Highway(c, 3, 3, False),
            l7=L.DeconvolutionND(1, c, c, 2, 2, initialW=chainer.initializers.HeNormal()),
            l8=Highway(c, 3, 1, False),
            l9=Highway(c, 3, 3, False),
            l10=Conv(c, 2*c, 1, 1, False),
            l11=Highway(2*c, 3, 1, False),
            l12=Highway(2*c, 3, 1, False),
            l13=Conv(2*c, fd, 1, 1, False),
            l14=Conv(fd, fd, 1, 1, False),
            l15=Conv(fd, fd, 1, 1, False),
            l16=Conv(fd, fd, 1, 1, False),
        )

    def __call__(self, x):
        h = x
        for i in range(1, 14):
            h = self['l%d'%i](h)
        h = F.leaky_relu(self.l14(h))
        h = F.leaky_relu(self.l15(h))
        h = F.leaky_relu(self.l16(h))
        return h

class SSRNNetwork(chainer.Chain):
    def __init__(self, f=80, c=512, fd=513):
        super(SSRNNetwork, self).__init__(
            ssrn=SSRN(f, c, fd),
        )

    def gen(self, x):
        y = self.ssrn(x)
        return y.data
        
    def __call__(self, x, t):
        y = self.ssrn(x)

        loss_l1 = F.mean_absolute_error(t, y)
        loss_bin = F.mean(F.bernoulli_nll(t, y, 'no'))

        loss = loss_l1 + loss_bin

        chainer.reporter.report(
            {
                'loss_l1': loss_l1,
                'loss_bin': loss_bin,
            }
        )
        
        return loss, y
        
