# sys
import h5py
import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from utils import common

# torch
import torch
from torchvision import datasets, transforms


def load_data(_path, _ftype_real, _ftype_synth, coords, joints, cycles=3):

    file_feature_real = os.path.join(_path, 'features' + _ftype_real + '.h5')
    ffr = h5py.File(file_feature_real, 'r')
    file_label_real = os.path.join(_path, 'labels' + _ftype_real + '.h5')
    flr = h5py.File(file_label_real, 'r')
    file_feature_synth = os.path.join(_path, 'features' + _ftype_synth + '.h5')
    ffs = h5py.File(file_feature_synth, 'r')
    file_label_synth = os.path.join(_path, 'labels' + _ftype_synth + '.h5')
    fls = h5py.File(file_label_synth, 'r')

    data_list = []
    num_samples_real = len(ffr.keys())
    num_samples_synth = len(ffs.keys())
    num_samples = num_samples_real + num_samples_synth
    time_steps = 0
    labels_real = np.empty(num_samples_real)
    labels_synth = np.empty(num_samples_synth)
    for si in range(num_samples_real):
        ffr_group_key = list(ffr.keys())[si]
        data_list.append(list(ffr[ffr_group_key]))  # Get the data
        time_steps_curr = len(ffr[ffr_group_key])
        if time_steps_curr > time_steps:
            time_steps = time_steps_curr
        labels_real[si] = flr[list(flr.keys())[si]][()]
    for si in range(num_samples_synth):
        ffs_group_key = list(ffs.keys())[si]
        data_list.append(list(ffs[ffs_group_key]))  # Get the data
        time_steps_curr = len(ffs[ffs_group_key])
        if time_steps_curr > time_steps:
            time_steps = time_steps_curr
        labels_synth[si] = fls[list(fls.keys())[si]][()]
    labels = np.concatenate((labels_real, labels_synth), axis=0)

    data = np.empty((num_samples, time_steps*cycles, joints*coords))
    for si in range(num_samples):
        data_list_curr = np.tile(data_list[si], (int(np.ceil(time_steps / len(data_list[si]))), 1))
        for ci in range(cycles):
            data[si, time_steps * ci:time_steps * (ci + 1), :] = data_list_curr[0:time_steps]
    data = common.get_affective_features(np.reshape(data, (data.shape[0], data.shape[1], joints, coords)))[:, :, :48]
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.1)
    return data, labels, data_train, labels_train, data_test, labels_test


def scale(_data):
    data_scaled = _data.astype('float32')
    data_max = np.max(data_scaled)
    data_min = np.min(data_scaled)
    data_scaled = (_data-data_min)/(data_max-data_min)
    return data_scaled, data_max, data_min


# descale generated data
def descale(data, data_max, data_min):
    data_descaled = data*(data_max-data_min)+data_min
    return data_descaled


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


class TrainTestLoader(torch.utils.data.Dataset):

    def __init__(self, data, label, joints, coords, num_classes):
        # data: N C T J
        self.data = np.reshape(data, (data.shape[0], data.shape[1], joints, coords, 1))
        self.data = np.moveaxis(self.data, [1, 2, 3], [2, 3, 1])

        # load label
        self.label = label

        self.N, self.C, self.T, self.J, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # processing
        # if self.random_choose:
        #     data_numpy = tools.random_choose(data_numpy, self.window_size)
        # elif self.window_size > 0:
        #     data_numpy = tools.auto_pading(data_numpy, self.window_size)
        # if self.random_move:
        #     data_numpy = tools.random_move(data_numpy)

        return data_numpy, label
