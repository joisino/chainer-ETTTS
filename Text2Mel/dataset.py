import os

import numpy as np
import chainer

class SynthesisDataset(chainer.dataset.DatasetMixin):
    def __init__(self, directory):
        self.directory = directory
        allfiles = os.listdir(directory)
        self.bases = []
        for f in allfiles:
            if f[-8:] == '_mel.npy':
                self.bases.append(f[:-8])
        self.wavs = []
        self.texts = []
        for i in range(len(self.bases)):
            filename = os.path.join(self.directory, self.bases[i] + '_mel.npy')
            wav = np.load(filename).astype('f')
            self.wavs.append(wav)

            filename = os.path.join(self.directory, self.bases[i] + '_txt.npy')
            text = np.load(filename).astype('i')
            self.texts.append(text)

    def __len__(self):
        return len(self.bases)

    def get_example(self, i):
        t = self.wavs[i]
        x = np.zeros((t.shape[0], 1)).astype('f')
        x = np.concatenate((x, t[:, :-1]), axis=1)
        text = self.texts[i]
        
        return text, x, t
