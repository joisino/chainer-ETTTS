#!/usr/bin/env python3

import os
import argparse
import string

import numpy as np
import librosa
from pykakasi import kakasi

kakasi = kakasi()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str)
    parser.add_argument('out', type=str)
    parser.add_argument('--gamma', type=float, default=0.6)
    args = parser.parse_args()

    chars = string.ascii_lowercase + ',.- \"'

    def check(s):
        for c in s:
            if c not in chars:
                return False
        return True

    kakasi.setMode('H', 'a')
    kakasi.setMode('K', 'a')
    kakasi.setMode('J', 'a')
    conv = kakasi.getConverter()

    def convert(s):
        res = conv.do(s)
        res = res.replace('、', ', ')
        res = res.replace('。', '.')
        return res

    metafile = os.path.join(args.root, 'transcript_utf8.txt')

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    with open(metafile) as f:
        for x in f:
            li = x[:-1].split(':')
            basename = li[0]
            text = convert(li[1])
            valid = check(text)
            print(basename, text, check(text))
            if valid:
                # convert wav file
                filename = os.path.join(args.root, 'wav', basename + '.wav')
                x, rate = librosa.load(filename)
                x = librosa.resample(x, rate, 22050)
                y = librosa.feature.melspectrogram(x, n_fft=1024, hop_length=256, n_mels=80)
                y = y[:, 3::4]
                y = (y/np.max(y)) ** args.gamma
                y = y.astype('f')

                t = librosa.core.stft(x, n_fft=1024, hop_length=256)
                t = np.abs(t)
                t = t[:, :t.shape[1] // 4 * 4]
                t = (t/np.max(t)) ** args.gamma
                t = t.astype('f')

                filename = os.path.join(args.out, basename + '_mel')
                np.save(filename, y)

                filename = os.path.join(args.out, basename + '_fft')
                np.save(filename, t)

                # convert txt file
                li = []
                for c in text:
                    li.append(chars.index(c))
                x = np.array(li, dtype='i')

                filename = os.path.join(args.out, basename + '_txt')
                np.save(filename, x)

if __name__ == '__main__':
    main()
