#!/usr/bin/env python3

import argparse
import os
import subprocess

import chainer
from chainer import iterators, optimizers, serializers
from chainer import training
from chainer.training import extensions

import dataset
import network
from updater import SynthesisUpdater

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--dump', '-d', type=str, default=None)
    parser.add_argument('--out', '-o', type=str, default=None)
    parser.add_argument('--epoch', '-e', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--batch', '-b', type=int, default=1)
    parser.add_argument('--base', type=str, default='.')
    parser.add_argument('--data', type=str, required=True)
    args = parser.parse_args()

    train = dataset.SynthesisDataset(args.data)
    train_iter = iterators.SerialIterator(train, batch_size=args.batch)

    model = network.SynthesisNetwork()

    opt = optimizers.Adam(alpha=args.alpha, beta1=args.beta1, beta2=args.beta2)
    opt.setup(model)

    opt.add_hook(chainer.optimizer.GradientClipping(1.0))

    updater = SynthesisUpdater(
        net=model,
        iterator={'main': train_iter},
        optimizer={'opt': opt},
        device=args.gpu,
    )

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=os.path.join(args.base, 'results'))

    trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
    trainer.extend(extensions.snapshot(filename='dump'), trigger=(100, 'epoch'))
    trainer.extend(extensions.PlotReport(['loss_bin'], trigger=(1, 'epoch'), file_name='loss_bin.png'))
    trainer.extend(extensions.PlotReport(['loss_l1'], trigger=(1, 'epoch'), file_name='loss_l1.png'))
    trainer.extend(extensions.PlotReport(['loss_att'], trigger=(1, 'epoch'), file_name='loss_att.png'))
    trainer.extend(extensions.PrintReport(['epoch', 'loss_bin', 'loss_l1', 'loss_att']))
    trainer.extend(extensions.ProgressBar(update_interval=1))

    if args.dump:
        print('loading dump from ' + args.dump)
        serializers.load_npz(args.dump, trainer)

    print('saving model to ' + args.out)
    serializers.save_npz(args.out, trainer.updater.net)

if __name__ == '__main__':
    main()
