import numpy as np
import random
import math
import warnings
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')


def extract_data_and_labels(features, labels):
	sample_features = []
	sample_labels = []
	for x in features:
		sample_features.append(x[1:])
		sample_labels.append(labels[x[0]].value)
	data_x = np.array(sample_features)
	data_y = np.array(sample_labels)
	return data_x, data_y


def cross_validate(features, labels):
	n = 1000
	total_error = 0
	for count in range(n):
		random.shuffle(features)
		num_features = len(features)
		training_features = features[int(math.floor(num_features/10)):]
		testing_features = features[:int(math.floor(num_features/10))]
		
		data_x, data_y = extract_data_and_labels(training_features, labels)
		# model = SVC()
		model = RandomForestClassifier()
		model.fit(data_x, data_y)

		data_x, data_y = extract_data_and_labels(testing_features, labels)

		pred_y = model.predict(data_x)
		error = pred_y - data_y
		e = np.count_nonzero(error)*1.0/data_y.shape[0]
		total_error += e * 100

	print('{0:.2f}'.format(100 - total_error/n))
