import chainer
import matplotlib.pyplot as plt

class SSRNUpdater(chainer.training.StandardUpdater):
    def __init__(self, **kwargs):
        self.net = kwargs.pop('net')
        self.cnt = 0
        super(SSRNUpdater, self).__init__(**kwargs)
        
    def update_core(self):
        opt = self.get_optimizer('opt')
        xp = self.net.xp

        data = self.get_iterator('main').next()
        x = []
        t = []
        for b in data:
            cx, ct = b
            x.append(cx)
            t.append(ct)
        x = xp.array(x, dtype=xp.float32)
        t = xp.array(t, dtype=xp.float32)

        loss, y= self.net(x, t)

        self.net.cleargrads()
        loss.backward()
        opt.update()

        if self.cnt//1000 != (self.cnt+1)//1000:
            def visualize(filename, img):
                img = chainer.cuda.to_cpu(img)
                plt.pcolor(img)
                plt.savefig(filename)
                plt.close()
            visualize('results/x.png', x[0])
            visualize('results/t.png', t[0])
            visualize('results/y.png', y.data[0])
        self.cnt += 1
