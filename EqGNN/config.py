from EqGNN.constants import LOG_PATH

import argparse

parser = argparse.ArgumentParser()


# general / saving constants #
##############################
parser.add_argument('--dataset', type=str, default='pokec', choices=['pokec', 'NBA'], help='the dataset name')
parser.add_argument('--sensitive', type=str, default='gender',
                    choices=['gender', 'region', None], dest='sensitive_attribute',
                    help='relevant only for pokec dataset: sensitive attribute name')
parser.add_argument('--seed', type=int, default=131, help='seed for data splitting randomness')
parser.add_argument('--log_path', type=str, default=LOG_PATH, help='tensorboard log path')
parser.add_argument('--gpus', type=str, default='0', help='gpus parameter used for pytorch_lightning')
# training constants #
######################
parser.add_argument('--lr', type=float, default=1e-3, dest='learning_rate', help='learning rate')
parser.add_argument('--wd', type=float, default=1e-5, dest='weight_decay', help='weight_decay')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout for the classifier')
parser.add_argument('--dim', type=int, default=128, dest='embedding_size', help='the desired embedding size')
parser.add_argument('--epochs', type=int, default=1000, help='number of maximum epochs to run')
parser.add_argument('--lmb', type=float, default=1, help='weight of the discriminator')
parser.add_argument('--gamma', type=float, default=50, help='weight of the covariance')
parser.add_argument('--loss', type=str, default='permutation', dest='discriminator_loss',
                    choices=['permutation', 'paired', 'unpaired', 'FairGNN', 'Debias', 'None'],
                    help='the desired loss of the discriminator')
parser.add_argument('--use_hidden', action='store_true', help='use hidden representation as discriminator input')
parser.add_argument('--NN_discriminator', action='store_true', help='use NN discriminator instead of GNN')
parser.add_argument('--graph_sampler', action='store_true', help='use graph sampler instead of bayes\' rule')

