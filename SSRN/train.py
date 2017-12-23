#!/usr/bin/env python3

import argparse
import subprocess
import os

import chainer
from chainer import iterators, optimizers, serializers
from chainer import training
from chainer.training import extensions
from chainer import cuda
import matplotlib

matplotlib.use('Agg')

import dataset
import network
from updater import SSRNUpdater

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--dump', '-d', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--opt', type=str, default=None)
    parser.add_argument('--epoch', '-e', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--batch', '-b', type=int, default=16)
    parser.add_argument('--base', type=str, default='.')
    parser.add_argument('--data', type=str, required=True)
    args = parser.parse_args()

    train = dataset.SSRNDataset(args.data)
    train_iter = iterators.SerialIterator(train, batch_size=args.batch)

    model = network.SSRNNetwork()

    if args.model:
        print('loading model from ' + args.model)
        serializers.load_npz(args.model, model)

    if args.gpu >= 0:
        cuda.get_device_from_id(0).use()
        model.to_gpu()

    opt = optimizers.Adam(alpha=args.alpha, beta1=args.beta1, beta2=args.beta2)

    opt.setup(model)
    opt.add_hook(chainer.optimizer.GradientClipping(1.0))

    if args.opt:
        print('loading opt from ' + args.opt)
        serializers.load_npz(args.opt, opt)

    updater = SSRNUpdater(
        net=model,
        iterator={'main': train_iter},
        optimizer={'opt': opt},
        device=args.gpu,
    )

    dirname = os.path.join(args.base, 'results')
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=dirname)

    trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
    trainer.extend(extensions.snapshot(filename='dump'), trigger=(10, 'epoch'))
    trainer.extend(extensions.PlotReport(['loss_l1'], trigger=(1, 'epoch'), file_name='loss_l1.png'))
    trainer.extend(extensions.PlotReport(['loss_bin'], trigger=(1, 'epoch'), file_name='loss_bin.png'))
    trainer.extend(extensions.PrintReport(['epoch', 'loss_l1', 'loss_bin']))
    trainer.extend(extensions.ProgressBar(update_interval=1))

    if args.dump:
        print('loading dump from ' + args.dump)
        serializers.load_npz(args.dump, trainer)

    trainer.run()

if __name__ == '__main__':
    main()
