import argparse
import os
import numpy as np
from utils import loader, processor

import torch
import torchlight


base_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(base_path, '../data')
ftype_real = ''
ftype_synth = 'CVAEGCN_best'
coords = 3
joints = 16
cycles = 1
model_path = os.path.join(base_path, 'model_classifier_stgcn/features'+ftype_real+'+features'+ftype_synth)


parser = argparse.ArgumentParser(description='Gait Gen')
parser.add_argument('--train', type=bool, default=True, metavar='T',
                    help='train the model (default: True)')
parser.add_argument('--save-features', type=bool, default=True, metavar='SF',
                    help='save penultimate layer features (default: True)')
parser.add_argument('--batch-size', type=int, default=8, metavar='B',
                    help='input batch size for training (default: 8)')
parser.add_argument('--num-worker', type=int, default=4, metavar='W',
                    help='input batch size for training (default: 4)')
parser.add_argument('--start_epoch', type=int, default=0, metavar='SE',
                    help='starting epoch of training (default: 0)')
parser.add_argument('--num_epoch', type=int, default=500, metavar='NE',
                    help='number of epochs to train (default: 500)')
parser.add_argument('--optimizer', type=str, default='Adam', metavar='O',
                    help='optimizer (default: Adam)')
parser.add_argument('--base-lr', type=float, default=0.1, metavar='L',
                    help='base learning rate (default: 0.1)')
parser.add_argument('--step', type=list, default=[0.5, 0.75, 0.875], metavar='[S]',
                    help='fraction of steps when learning rate will be decreased (default: [0.5, 0.75, 0.875])')
parser.add_argument('--nesterov', action='store_true', default=True,
                    help='use nesterov')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='D',
                    help='Weight decay (default: 5e-4)')
parser.add_argument('--eval-interval', type=int, default=1, metavar='EI',
                    help='interval after which model is evaluated (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='LI',
                    help='interval after which log is printed (default: 100)')
parser.add_argument('--topk', type=list, default=[1], metavar='[K]',
                    help='top K accuracy to show (default: [1])')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--pavi-log', action='store_true', default=False,
                    help='pavi log')
parser.add_argument('--print-log', action='store_true', default=True,
                    help='print log')
parser.add_argument('--save-log', action='store_true', default=True,
                    help='save log')
parser.add_argument('--work-dir', type=str, default=model_path, metavar='WD',
                    help='path to save')
# TO ADD: save_result

args = parser.parse_args()
device = 'cuda:0'

data, labels, data_train, labels_train, data_test, labels_test =\
    loader.load_data(data_path, ftype_real, ftype_synth, coords, joints, cycles=cycles)
num_classes = np.unique(labels_train).shape[0]
data_loader = {
    'train': torch.utils.data.DataLoader(
        dataset=loader.TrainTestLoader(data_train, labels_train, joints, coords, num_classes),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True),
    'test': torch.utils.data.DataLoader(
        dataset=loader.TrainTestLoader(data_test, labels_test, joints, coords, num_classes),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True)}
graph_dict = {'strategy': 'spatial'}
print('Train set size: {:d}'.format(len(data_train)))
print('Test set size: {:d}'.format(len(data_test)))
print('Number of classes: {:d}'.format(num_classes))
pr = processor.Processor(args, data_loader, coords, num_classes, graph_dict, device=device)
if args.train:
    pr.train()
if args.save_features:
    f = pr.save_best_feature(ftype_real, ftype_synth, data, joints, coords)
