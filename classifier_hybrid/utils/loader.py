# sys
import h5py
import math
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# torch
import torch
from torchvision import datasets, transforms


def load_data(_path, _ftype, joints, coords, cycles=1, test_size=0.1):

    file_affeature = os.path.join(_path, 'affectiveFeatures'+_ftype+'.h5')
    ff1 = h5py.File(file_affeature, 'r')

    file_gait = os.path.join(_path, 'features' + _ftype + '.h5')
    ff2 = h5py.File(file_gait, 'r')

    file_label = os.path.join(_path, 'labels' + _ftype + '.h5')
    ff3 = h5py.File(file_label, 'r')

    aff_list = []
    num_samples = len(ff1.keys())
    for si in range(num_samples):
        ff1_group_key = list(ff1.keys())[si]
        aff_list.append(ff1[ff1_group_key])
    aff = np.array(aff_list)

    gait_list = []
    num_samples = len(ff2.keys())
    time_steps = 0
    labels = np.empty(num_samples)
    for si in range(num_samples):
        ff2_group_key = list(ff2.keys())[si]
        gait_list.append(list(ff2[ff2_group_key]))  # Get the data
        time_steps_curr = len(ff2[ff2_group_key])
        if time_steps_curr > time_steps:
            time_steps = time_steps_curr
        labels[si] = ff3[list(ff3.keys())[si]][()]

    gait = np.empty((num_samples, time_steps*cycles, joints*coords))
    for si in range(num_samples):
        gait_list_curr = np.tile(gait_list[si], (int(np.ceil(time_steps / len(gait_list[si]))), 1))
        for ci in range(cycles):
            gait[si, time_steps * ci:time_steps * (ci + 1), :] = gait_list_curr[0:time_steps]

    data = [(a, g) for a, g in zip(aff, gait)]
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=test_size)
    return data_train, data_test, labels_train, labels_test


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

    def __init__(self, data, label, joints, coords):
        # load data
        self.aff = []
        self.gait = []
        for aff, gait in data:
            self.aff.append(aff)
            self.gait.append(gait)
        self.aff = np.array(self.aff)
        self.gait = np.array(self.gait)
        self.gait = np.reshape(self.gait, (self.gait.shape[0], self.gait.shape[1], joints, coords, 1))
        self.gait = np.moveaxis(self.gait, [1, 2, 3], [2, 3, 1])

        # load label
        self.label = label

        self.F = self.aff.shape[1]
        self.N, self.C, self.T, self.J, self.M = self.gait.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        aff = np.array(self.aff[index])
        gait = np.array(self.gait[index])
        label = self.label[index]
        return aff, gait, label
