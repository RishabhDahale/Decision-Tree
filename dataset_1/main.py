import pandas as pd
import numpy as np
import csv
import sys

dataset=pd.read_csv(sys.argv[2])
mean_data = np.mean(dataset.iloc[:,-1])
test_data = pd.read_csv(sys.argv[4])

def variance(data, attribute_name, target_name="output"):
	feature_value = np.unique(data[attribute_name])
	feature_variance = 0
	for value in feature_value:
		subset = data.query('{0}=={1}'.format(attribute_name, value)).reset_index()
		value_var = (len(subset)/len(data))*np.var(subset[target_name], ddof=1)
		feature_variance+=value_var

	return feature_variance

def Classification(data, original_data, features, min_instances, target_name, parent_node_class=None):

	if len(data) <= int(min_instances):
		return np.mean(data[target_name])
	elif len(data) == 0:
		return np.mean(original_data[target_name])
	elif len(features)==0:
		return parent_node_class
	else:
		parent_node_class = np.mean(data[target_name])

		item_values = [variance(data, feature) for feature in features]
		bestFeatureIndex = np.argmin(item_values)
		bestFeature = features[bestFeatureIndex]

		tree = {bestFeature:{}}

		features = [i for i in features if i != bestFeature]

		for value in np.unique(data[bestFeature]):
			value = value

			sub_data = data.where(data[bestFeature] == value).dropna()

			subTree = Classification(sub_data, original_data, features, min_instances, 'output', parent_node_class = parent_node_class)

			tree[bestFeature][value] = subTree
		return tree

def predict(test, tree, default = mean_data):
	for key in list(test.keys()):
		if key in list(tree.keys()):
			try:
				result = tree[key][test[key]]
				# print(result)
			except:
				return default
			result = tree[key][test[key]]
			if isinstance(result, dict):
				return predict(test, result)
			else:
				# print(result)
				# print()
				return result

def train_test_split(dataset):
	train_data = dataset.iloc[:int(len(dataset))].reset_index(drop = True)
	# test_data = dataset.iloc[(int(0.8*len(dataset))):].reset_index(drop = True)
	return train_data

dataset= train_test_split(dataset)
# test_data = dataset[1]
# dataset = dataset[0]
test_data = train_test_split(test_data)

def test(data, tree):
	queries = data.iloc[:,:-1].to_dict(orient = "records")
	predicted = []
	for i in range(len(data)):
		predicted.append(predict(queries[i], tree, mean_data))

	# print(predicted)
	with open('output.csv','w',newline='') as csvfile:
		writerr = csv.writer(csvfile,delimiter=',')
		writerr.writerow(['id','output'])
		ii = 0
		for p in predicted:
			ii += 1
			# xid = int(temp[i+1,0])
            # p = int(p)
			writerr.writerow([ii,p])

tree = Classification(dataset, dataset, dataset.columns[:-1], 20, 'output')
test(test_data, tree)