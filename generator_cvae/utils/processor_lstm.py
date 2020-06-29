import h5py
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from net import CVAE_lstm as CVAE
from utils import loader_lstm as loader
from utils import losses
from utils.common import *
from torchlight import torchlight


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def vae_loss(x_in, x_out, mean, lsig, beta=1.):
    BCE = nn.functional.binary_cross_entropy(x_out, x_in)
    KLD = -0.5 * torch.sum(1 + lsig - mean.pow(2) - lsig.exp())
    return BCE + beta*KLD


def get_best_epoch(path_to_model_files):
    all_models = os.listdir(path_to_model_files)
    while '_' not in all_models[-1]:
        all_models = all_models[:-1]
    best_model = all_models[-1]
    return int(best_model[5:best_model.find('_')])


class Processor(object):
    """
        Processor for gait generation
    """

    def __init__(self, args, ftype, data_loader, data_max, data_min, C, T, V, F, num_classes, n_z=1024, device='cuda:0'):

        self.args = args
        self.ftype = ftype
        self.data_loader = data_loader
        self.data_max = data_max
        self.data_min = data_min
        self.num_classes = num_classes
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)
        self.device = device
        self.io = torchlight.IO(
            self.args.work_dir,
            save_log=self.args.save_log,
            print_log=self.args.print_log)

        # model
        self.C = C
        self.T = T
        self.V = V
        self.F = F
        self.n_z = n_z
        if not os.path.isdir(self.args.work_dir):
            os.mkdir(self.args.work_dir)
        self.model = CVAE.CVAE(F, T, self.n_z, num_classes)
        self.model.cuda('cuda:0')
        self.model.apply(weights_init)
        self.loss = vae_loss
        self.best_loss = np.inf
        self.loss_updated = False
        self.step_epochs = [np.ceil(float(self.args.num_epoch * x)) for x in self.args.step]
        self.best_epoch = None
        self.mean = 0.
        self.lsig = 1.

        # optimizer
        if self.args.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.base_lr,
                momentum=0.9,
                nesterov=self.args.nesterov,
                weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.base_lr,
                weight_decay=self.args.weight_decay)
        else:
            raise ValueError()
        self.lr = self.args.base_lr

    def adjust_lr(self):

        # if self.args.optimizer == 'SGD' and\
        if self.meta_info['epoch'] in self.step_epochs:
            lr = self.args.base_lr * (
                    0.1 ** np.sum(self.meta_info['epoch'] >= np.array(self.step_epochs)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr

    def show_epoch_info(self):

        print_epoch = self.best_epoch if self.best_epoch is not None else 0
        for k, v in self.epoch_info.items():
            self.io.print_log('\t{}: {:.4f}. Best so far: {:.4f} (epoch {:d}).'.
                              format(k, v, self.best_loss, print_epoch))
        if self.args.pavi_log:
            self.io.log('train', self.meta_info['iter'], self.epoch_info)

    def show_iter_info(self):

        if self.meta_info['iter'] % self.args.log_interval == 0:
            info = '\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            self.io.print_log(info)

            if self.args.pavi_log:
                self.io.log('train', self.meta_info['iter'], self.iter_info)

    def per_train(self):

        self.model.train()
        self.adjust_lr()
        train_loader = self.data_loader['train']
        loss_value = []

        for data, label in train_loader:
            # get data
            data = data.float().to(self.device)
            ldec = label.float().to(self.device)
            lenc = ldec.unsqueeze(1).repeat([1, data.shape[1], 1])

            # forward
            output, self.mean, self.lsig, _ = self.model(data, lenc, ldec)
            loss = self.loss(data, output, self.mean, self.lsig)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        temp = output.detach().cpu().numpy()[0, :, :]
        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def per_test(self, evaluation=True):

        self.model.eval()
        test_loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for data, label in test_loader:

            # get data
            data = data.float().to(self.device)
            ldec = label.float().to(self.device)
            lenc = ldec.unsqueeze(1).repeat([1, data.shape[1], 1])

            # inference
            with torch.no_grad():
                output, mean, lsig, _ = self.model(data, lenc, ldec)
            result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(data, output, mean, lsig)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss'] = np.mean(loss_value)
            if self.epoch_info['mean_loss'] < self.best_loss:
                self.best_loss = self.epoch_info['mean_loss']
                self.best_epoch = self.meta_info['epoch']
                self.loss_updated = True
            else:
                self.loss_updated = False
            self.show_epoch_info()

    def train(self):

        for epoch in range(self.args.start_epoch, self.args.num_epoch):
            self.meta_info['epoch'] = epoch

            # training
            self.io.print_log('Training epoch: {}'.format(epoch))
            self.per_train()
            self.io.print_log('Done.')

            # evaluation
            if (epoch % self.args.eval_interval == 0) or (
                    epoch == self.args.num_epoch):
                self.io.print_log('Eval epoch: {}'.format(epoch))
                self.per_test()
                self.io.print_log('Done.')

            # save model and weights
            if self.loss_updated:
                torch.save(self.model.state_dict(),
                           os.path.join(self.args.work_dir, 'epoch{}_model.pth.tar'.format(epoch)))
                self.generate(epoch=str(epoch))

    def test(self):

        # the path of weights must be appointed
        if self.args.weights is None:
            raise ValueError('Please appoint --weights.')
        self.io.print_log('Model:   {}.'.format(self.args.model))
        self.io.print_log('Weights: {}.'.format(self.args.weights))

        # evaluation
        self.io.print_log('Evaluation Start:')
        self.per_test()
        self.io.print_log('Done.\n')

        # save the output of model
        if self.args.save_result:
            result_dict = dict(
                zip(self.data_loader['test'].dataset.sample_name,
                    self.result))
            self.io.save_pkl(result_dict, 'test_result.pkl')

    # def generate(self, base_path, data_max, data_min, max_z=1.5, total_samples=10, fill=5):
    def generate(self, max_z=1.5, total_samples=10, fill=5, epoch=''):
        # load model
        if self.best_epoch is None:
            self.best_epoch = get_best_epoch(self.args.work_dir)
        filename = os.path.join(self.args.work_dir, 'epoch{}_model.pth.tar'.format(self.best_epoch))
        self.model.load_state_dict(torch.load(filename))

        emotions = ['Angry', 'Neutral', 'Happy', 'Sad']
        ffile = 'features'+self.ftype+'CVAELSTM'
        ffile += '_'+epoch+'.h5' if epoch else '.h5'
        lfile = 'labels'+self.ftype+'CVAELSTM'
        lfile += '_'+epoch+'.h5' if epoch else '.h5'
        h5Featr = h5py.File(os.path.join(self.args.data_dir, ffile), 'w')
        h5Label = h5py.File(os.path.join(self.args.data_dir, lfile), 'w')
        for count in range(total_samples):
            gen_seqs = np.empty((self.num_classes, self.T, self.V*self.C))
            for cls in range(self.num_classes):
                ldec = np.zeros((1, self.num_classes), dtype='float32')
                ldec[0, cls] = 1.
                # z = np.zeros((1, self.n_z), dtype='float32')
                # z[0, 0] = np.random.random_sample() * max_z * 2 - max_z
                # z[0, 1] = np.random.random_sample() * max_z * 2 - max_z
                z = np.float32(np.random.randn(1, self.n_z))
                with torch.no_grad():
                    z = to_var(torch.from_numpy(z))
                    ldec = to_var(torch.from_numpy(ldec))
                    gen_seq_curr = self.model.decoder(z, ldec, self.T)
                    gen_seqs[cls, :, :] = gen_seq_curr.cpu().numpy()[:, :, :self.V*self.C]
            for idx in range(gen_seqs.shape[0]):
                h5Featr.create_dataset(str(count + 1).zfill(fill) + '_' + emotions[idx],
                                       data=loader.descale(gen_seqs[idx, :, :], self.data_max, self.data_min))
                # h5Featr.create_dataset(str(count + 1).zfill(fill) + '_' + emotions[idx], data=gen_seqs[idx, :, :])
                h5Label.create_dataset(str(count + 1).zfill(fill) + '_' + emotions[idx], data=idx)
            print('\rGenerating data: {:d} of {:d} ({:.2f}%).'
                  .format(count+1, total_samples, 100*(count+1)/total_samples), end='')
        h5Featr.close()
        h5Label.close()
        print()
