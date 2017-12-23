import os

import numpy as np
import chainer

class SSRNDataset(chainer.dataset.DatasetMixin):
    def __init__(self, directory):
        super(SSRNDataset, self).__init__()
        self.directory = directory
        allfiles = os.listdir(directory)
        self.bases = []
        for f in allfiles:
            if f[-8:] == '_mel.npy':
                self.bases.append(f[:-8])

        self.x = []
        self.t = []
        for b in self.bases:
            filename = os.path.join(self.directory, b + '_mel.npy')
            self.x.append(np.load(filename).astype('f'))

            filename = os.path.join(self.directory, b + '_fft.npy')
            self.t.append(np.load(filename).astype('f'))


    def __len__(self):
        return len(self.bases)

    def get_example(self, i):
        xlen = self.x[i].shape[1]
        if xlen >= 64:
            pos = np.random.randint(self.x[i].shape[1]-64+1)

            x = self.x[i][:, pos:pos+64]
            t = self.t[i][:, pos*4:pos*4+256]
        else:
            x = np.pad(self.x[i], ((0, 0), (0, 64-xlen)), 'constant', constant_values=0)
            t = np.pad(self.t[i], ((0, 0), (0, (64-xlen)*4)), 'constant', constant_values=0)

        return x, t
