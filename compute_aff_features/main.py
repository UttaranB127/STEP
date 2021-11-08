import h5py
import os

# from compute_features import compute_features
# from normalize_features import normalize_features
# from cross_validate import cross_validate
from compute_aff_features.compute_features import compute_features
from compute_aff_features.normalize_features import normalize_features
from compute_aff_features.cross_validate import cross_validate


base_path = '/mnt/q/Gamma/Gait'

# Get affective features
f_type = ''
positions = h5py.File(os.path.join(base_path, 'data', 'features{}.h5'.format(f_type)), 'r')
keys = positions.keys()
time_step = 1.0 / 30.0

print('Computing Features ... ', end='')
features = []
for key in keys:
	frames = positions[key]
	feature = [key]
	if frames.ndim == 2:
		feature += compute_features(frames, time_step)
	else:
		feature += frames
	features.append(feature)

print('done.\nNormalizing Features ... ', end='')
normalized_features = []
normalize_features(features, normalized_features)
print('done.\nSaving features ... ', end='')
with h5py.File('../../data/affectiveFeatures{}.h5'.format(f_type), 'w') as aff:
	for feature in normalized_features:
		aff.create_dataset(feature[0], data=feature[1:])
print('done.')

# Get labels
labels = h5py.File(os.path.join(base_path, 'data', 'labels{}.h5'.format(f_type)), 'r')

# Cross validate
cross_validate(normalized_features, labels)
