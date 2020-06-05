import h5py
import os
import numpy as np

base_path = os.path.dirname(os.path.realpath(__file__))
feature_file = '/media/uttaran/FCE1-7BF3/Gamma/Gait/classifier_stgcn/model_classifier_stgcn/featuresCombineddeep_features.txt'
f = np.loadtxt(feature_file)
fCombined = h5py.File('/media/uttaran/FCE1-7BF3/Gamma/Gait/data/featuresCombined.h5', 'r')
fkeys = fCombined.keys()
dfCombined = h5py.File('/media/uttaran/FCE1-7BF3/Gamma/Gait/data/deepFeaturesCombined.h5', 'w')
for i, fkey in enumerate(fkeys):
    fname = [fkey][0]
    feature = f[i, :]
    dfCombined.create_dataset(fname, data=feature)
dfCombined.close()
