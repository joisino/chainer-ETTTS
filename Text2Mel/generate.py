#!/usr/bin/env python3

import string
import argparse
import sys

import numpy as np
import chainer
from pykakasi import kakasi

kakasi = kakasi()

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import network

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--text', type=str, default='icassp stands for the international conference on acoustics, speech and signal processing.')
    parser.add_argument('--len', type=int, default=100)
    parser.add_argument('--model', '-m', type=str, required=True)
    parser.add_argument('--ja', '-j', action='store_true')
    args = parser.parse_args()

    chars = string.ascii_lowercase + ',.- \"'

    model = network.SynthesisNetwork()

    print('loading model from ' + args.model)
    chainer.serializers.load_npz(
        args.model,
        model,
    )
    print('model loaded')

    xp = np

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

        xp = chainer.cuda.cupy

    ctext = args.text
    if args.ja:
        kakasi.setMode("H","a")
        kakasi.setMode("K","a")
        kakasi.setMode("J","a")
        kakasi.setMode("r","Hepburn")
        kakasi.setMode("C", True)
        kakasi.setMode("c", False)
        conv = kakasi.getConverter()
        ctext = conv.do(ctext).lower()
        ctext = ctext.replace('、', ', ')
        ctext = ctext.replace('。', '.')

    print(ctext)

    li = []
    for c in ctext:
        li.append(chars.index(c))

    text = xp.array(li).astype('i')
    text = xp.expand_dims(text, 0)

    x = xp.zeros((1, 80, 1)).astype('f')
    cnt = args.len
    for i in range(args.len):
        sys.stdout.write('\r%d' % i)
        sys.stdout.flush()
        with chainer.using_config('train', False):
            y, a = model.gen(text, x)
        x = xp.concatenate((xp.zeros((1, 80, 1)).astype('f'), y), axis=2)

        cnt -= 1
        if xp.argmax(a[0, :, -1]) >= len(ctext) - 3:
            cnt = min(cnt, 10)

        if cnt <= 0:
            break

    sys.stdout.write('\n')
    sys.stdout.flush()

    img = chainer.cuda.to_cpu(y[0])

    plt.pcolor(img)
    plt.savefig('./results/gen.png')
    plt.close()

    img = chainer.cuda.to_cpu(a[0])

    plt.pcolor(img)
    plt.savefig('./results/gen_a.png')
    plt.close()

    y = chainer.cuda.to_cpu(y)

    np.save('./results/res.npy', y)

if __name__ == '__main__':
    main()
