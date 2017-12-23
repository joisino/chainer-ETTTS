#!/usr/bin/env python3

import argparse

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import chainer
import librosa
import lws

import network

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--mel', type=str, required=True)
    parser.add_argument('--gamma', type=float, default=0.6)
    parser.add_argument('--eta', type=float, default=1.3)
    args = parser.parse_args()

    model = network.SSRNNetwork()

    print('loading model from ' + args.model)
    chainer.serializers.load_npz(args.model, model)
    print('model loaded')

    y = np.load(args.mel)
    t = model.gen(y)
    t = t[0].astype(np.float64)
    t[t<0] = 0
    t = t ** (args.eta / args.gamma) * 50
    t = np.transpose(t, (1, 0))
    print(t.shape)
    plt.pcolor(t)
    plt.savefig('res.png')

    lws_processor=lws.lws(1024, 256, mode='speech', fftsize=1024)
    t = lws_processor.run_lws(t)
    t = lws_processor.istft(t)
    librosa.output.write_wav('./res.wav', t.astype('f'), 22050)

if __name__ == '__main__':
    main()
    
