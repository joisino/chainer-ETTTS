#!/usr/bin/env python3

import argparse

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
    parser.add_argument('--dump', '-d', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--epoch', '-e', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--batch', '-b', type=int, default=16)
    parser.add_argument('--data', type=str, default=None)
    args = parser.parse_args()

    train = dataset.SSRNDataset(args.data)
    train_iter = iterators.SerialIterator(train, batch_size=args.batch)

    model = network.SSRNNetwork()

    opt = optimizers.Adam(alpha=args.alpha, beta1=args.beta1, beta2=args.beta2)

    opt.setup(model)
    opt.add_hook(chainer.optimizer.GradientClipping(1.0))

    updater = SSRNUpdater(
        net=model,
        iterator={'main': train_iter},
        optimizer={'opt': opt},
        device=-1,
    )

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out='./results')

    trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
    trainer.extend(extensions.snapshot(filename='dump'), trigger=(10, 'epoch'))
    trainer.extend(extensions.PlotReport(['loss_l1'], trigger=(1, 'epoch'), file_name='loss_l1.png'))
    trainer.extend(extensions.PlotReport(['loss_bin'], trigger=(1, 'epoch'), file_name='loss_bin.png'))
    trainer.extend(extensions.PrintReport(['epoch', 'loss_l1', 'loss_bin']))
    trainer.extend(extensions.ProgressBar(update_interval=1))

    print('loading dump from ' + args.dump)
    serializers.load_npz(args.dump, trainer)

    print('saving train to ' + args.out)
    serializers.save_npz(args.out, trainer.updater.net)

if __name__ == '__main__':
    main()
    
