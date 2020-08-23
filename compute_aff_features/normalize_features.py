def normalize_features(features, normalized_features):
	# normalize features
	num_features = len(features[0])
	min_values = [float("inf")]*num_features
	max_values = [-float("inf")]*num_features
	for feature in features:
		for i in range(1, len(feature)):
			if min_values[i] > float(feature[i]):
				min_values[i] = float(feature[i])
			if max_values[i] < float(feature[i]):
				max_values[i] = float(feature[i])

	for feature in features:
		normalized_feature = [feature[0]]
		for i in range(1, len(feature)):
			a = (max_values[i] + min_values[i])/2
			b = (max_values[i] - min_values[i])/2
			if b == 0:
				normalized_feature.append(0)
			else:
				normalized_feature.append((float(feature[i]) - a)/b)
		normalized_features.append(normalized_feature)
