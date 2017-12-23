import numpy as np
import chainer
import matplotlib.pyplot as plt

class SynthesisUpdater(chainer.training.StandardUpdater):
    def __init__(self, **kwargs):
        self.net = kwargs.pop('net')
        self.cnt = 0
        super(SynthesisUpdater, self).__init__(**kwargs)

    def update_core(self):
        opt = self.get_optimizer('opt')
        xp = self.net.xp

        data = self.get_iterator('main').next()
        text = []
        x = []
        t = []
        textlens = []
        xlens = []
        for b in data:
            ctext, cx, ct = b
            textlens.append(ctext.shape[0])
            xlens.append(cx.shape[1])
        for b in data:
            ctext, cx, ct = b
            ctext = np.pad(ctext, (0, max(textlens)-ctext.shape[0]), 'constant', constant_values=31)
            cx = np.pad(cx, ((0, 0), (0, max(xlens)-cx.shape[1])), 'constant', constant_values=0)
            ct = np.pad(ct, ((0, 0), (0, max(xlens)-ct.shape[1])), 'constant', constant_values=0)
            text.append(ctext)
            x.append(cx)
            t.append(ct)
        text = xp.array(text, dtype=xp.int32)
        x = xp.array(x, dtype=xp.float32)
        t = xp.array(t, dtype=xp.float32)

        loss, y, a = self.net(text, x, t, textlens, xlens)

        self.net.cleargrads()
        loss.backward()
        opt.update()

        if self.get_iterator('main').is_new_epoch:
            def visualize(filename, img):
                img = chainer.cuda.to_cpu(img)
                plt.pcolor(img)
                plt.savefig(filename)
                plt.close()
            visualize('results/x.png', x[0, :, :xlens[0]])
            visualize('results/y.png', y.data[0, :, :xlens[0]])
            visualize('results/a.png', a.data[0, :textlens[0], :xlens[0]])
