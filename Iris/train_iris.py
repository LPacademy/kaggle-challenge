#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse

import numpy as np
from numpy import genfromtxt
from sklearn import datasets
from sklearn.model_selection import train_test_split

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions



class MLP(chainer.Chain):
    """
    Network definition
    """
    def __init__(self):
        super(MLP,self).__init__(
                l1 = L.Linear(4,100),
                l2 = L.Linear(100,100),
                l3 = L.Linear(100,3)
                )

    def __call__(self,x):    
         h = F.relu(self.l1(x))
         h = F.relu(self.l2(h))
         return self.l3(h)

def convert_onehot(data):
    y = np.array([int(i[0]) for i in data])
    y_onehot = [0] * len(y)
    for i, j in enumerate(y):
        y_onehot[i] = [0] * (y.max() + 1)
        y_onehot[i][j] = 1
    return (y, y_onehot)

def build_data():
    """
    The following shows 4 rows of orig dataset:

     SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species
     5.7,             2.9,           4.2,          1.3,         Iris-versicolor
     7.6,             3.0,           6.6,          2.1,         Iris-virginica
     5.6,             3.0,           4.5,          1.5,         Iris-versicolor
     5.1,             3.5,           1.4,          0.2,         Iris-setosa
     

    What we've done is placed the values 5.7, 2.9, ... up to but not 
    including the Species into x_train
    
       x_train[0:4]  ... 

           array([[ 5.7,  2.9,  4.2,  1.3], 
                  [ 7.6,  3. ,  6.6,  2.1],
                  [ 5.6,  3. ,  4.5,  1.5],
                  [ 5.1,  3.5,  1.4,  0.2]])


       y_train[0:4]  ...
     
           array([1, 2, 1, 0]) # 0 is Iris-setosa, 1 is Iris-versicolor, 2 is Iris-virginica


       y_train_onehot[0:4]

          [[0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]

       So now you have the following:
           [0, 1, 0]  represents 1, which is Iris-versicolor
           [0, 0, 1]  represents 2, which is Iris-virginica
           [0, 1, 0]  represents 1, which is Iris-versicolor (again)
           [1, 0, 0]  represents 0, which is Iris-setosa

        ...note our one hot encoding is actually flipped. [1,0,0] == 0
           [0,1,0] == 1
           [0,0,1] == 2
           That's okay, as long as it's consistent 
    """
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.33, random_state=42)
    f = open('cs-training.csv', 'w')
    for i, j in enumerate(X_train):
        k = np.append(np.array(y_train[i]), j)
        f.write(",".join([str(s) for s in k]) + '\n')
    f.close()
    f = open('cs-testing.csv', 'w')
    for i, j in enumerate(X_test):
        k = np.append(np.array(y_test[i]), j)
        f.write(",".join([str(s) for s in k]) + '\n')
    f.close()

def main():
    parser = argparse.ArgumentParser(description='Chainer example: Iris')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = L.Classifier(MLP())
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Write data to following files
    #   cs-traing.csv
    #   cs-testing.csv
    build_data()

    # load data from files
    train_data = genfromtxt('cs-training.csv',delimiter=',')
    test_data = genfromtxt('cs-testing.csv',delimiter=',')
    X_train = np.array([i[1:] for i in train_data])
    y_train, y_train_onehot = convert_onehot(train_data)
    X_test = np.array([i[1:] for i in test_data])
    y_train, y_train_onehot = convert_onehot(test_data)
    
    train = chainer.datasets.TupleDataset(X_train.T, y_train.T)
    test = chainer.datasets.TupleDataset(X_test.astype(np.float32), y_test.flatten().astype(np.int32))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at each epoch
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    trainer.extend(
        extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch',
                              file_name='loss.png'))
    trainer.extend(
        extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],
                              'epoch', file_name='accuracy.png'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

if __name__ == '__main__':
    main()
