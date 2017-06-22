# coding:utf-8

import math
import pickle
import copy
import numpy as np

model_dir = './model/'
model_name = 'model_'

tags = ['O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG']
indices = {'O':0, 'B-LOC':1, 'I-LOC':2, 'B-PER':3, 'I-PER':4, 'B-ORG':5, 'I-ORG':6}


# average structred perceptron
class StructuredPerceptron(object):
	# initialization
	def __init__(self):
		self.weights = {}      # weight of feature
		self.acc_weights = {}   # accumulate delta
		self.last_step = {}    # last step for updating weight
		self.step = 0          # current step

		# init weight
		# 'O' - 'I'
		self.weights['pred1+current'+tags[0]+tags[2]] = -float('inf')
		self.weights['pred1+current'+tags[0]+tags[4]] = -float('inf')
		self.weights['pred1+current'+tags[0]+tags[6]] = -float('inf')
		# 'B-LOC' - 'I-PER I-ORG'
		self.weights['pred1+current'+tags[1]+tags[4]] = -float('inf')
		self.weights['pred1+current'+tags[1]+tags[6]] = -float('inf')
		# 'B-PER' - 'I-LOC I-ORG'
		self.weights['pred1+current'+tags[3]+tags[2]] = -float('inf')
		self.weights['pred1+current'+tags[3]+tags[6]] = -float('inf')
		# 'B-ORG' - 'I-LOC I-PER'
		self.weights['pred1+current'+tags[5]+tags[2]] = -float('inf')
		self.weights['pred1+current'+tags[5]+tags[4]] = -float('inf')
		# 'I-LOC' - 'I-PER I-ORG'
		self.weights['pred1+current'+tags[2]+tags[4]] = -float('inf')
		self.weights['pred1+current'+tags[2]+tags[6]] = -float('inf')
		# 'I-PER' - 'I-LOC I-ORG'
		self.weights['pred1+current'+tags[4]+tags[2]] = -float('inf')
		self.weights['pred1+current'+tags[4]+tags[6]] = -float('inf')
		# 'I-ORG' - 'I-LOC I-PER'
		self.weights['pred1+current'+tags[6]+tags[2]] = -float('inf')
		self.weights['pred1+current'+tags[6]+tags[4]] = -float('inf')

		'''
		# punctuation
		for i in range(1, len(tags)):
			self.weights[tags[i]+'current，'] = -float('inf')
			self.weights[tags[i]+'current。'] = -float('inf')
			self.weights[tags[i]+'current、'] = -float('inf')
			self.weights[tags[i]+'current；'] = -float('inf')
			self.weights[tags[i]+'current！'] = -float('inf')
			self.weights[tags[i]+'current？'] = -float('inf')
			self.weights[tags[i]+'current《'] = -float('inf')
			self.weights[tags[i]+'current》'] = -float('inf')
			self.weights[tags[i]+'current“'] = -float('inf')
			self.weights[tags[i]+'current”'] = -float('inf')
			self.weights[tags[i]+'current（'] = -float('inf')
			self.weights[tags[i]+'current）'] = -float('inf')
			self.weights[tags[i]+'current『'] = -float('inf')
			self.weights[tags[i]+'current』'] = -float('inf')

		for feature in self.weights:
			self.acc_weights[feature] = -float('inf')
			self.last_step[feature] = 0
		'''

	# return the weight for a specific feature
	def get_weight(self, feature):
		return self.weights[feature] if feature in self.weights else 0

	# add delta to corresponding weight
	def update_weight(self, feature, delta):
		# if a feature not in record, create a new feature record
		if feature not in self.weights:
			self.weights[feature] = 0
			self.acc_weights[feature] = 0
			self.last_step[feature] = self.step
		else:
			# accumulate previous weight
			self.acc_weights[feature] += (self.step - self.last_step[feature]) * self.weights[feature]
			self.last_step[feature] = self.step

		self.weights[feature] += delta


	# add delta to every weight
	def update(self, X, Y, delta):
		length = len(Y)
		# node features
		for i, features in zip(range(length), self.generate_node_features(X)):
			for feature in features:
				self.update_weight(Y[i]+feature, delta)
		# edge features
		for features in self.generate_edge_features_update(Y):
			for feature in features:
				self.update_weight(feature, delta)

	# calculate the average weight
	def average(self):
		# update acc_weights
		for feature, acc_weight in self.acc_weights.items():
			if self.last_step[feature] != self.step:
				acc_weight += (self.step - self.last_step[feature]) * self.weights[feature]
			self.weights[feature] = acc_weight / self.step

	# node features
	def generate_node_features(self, X):
		length = len(X)
		for i in range(length):
			pred3 = X[i-3] if i-3 >= 0 else 'start'
			pred2 = X[i-2] if i-2 >= 0 else 'start'
			pred1 = X[i-1] if i-1 >= 0 else 'start'
			current = X[i]
			post1 = X[i+1] if i+1 < length else 'end'
			post2 = X[i+2] if i+2 < length else 'end'
			post3 = X[i+3] if i+3 < length else 'end'
			feature_vector = ['current'+current,
			                  'pred1'+pred1,
			                  'pred2'+pred2,
			                  'pred3'+pred3,
			                  'post1'+post1,
			                  'post2'+post2,
			                  'post3'+post3,
			                  'pred1+current'+pred1+current,
			                  'current+post1'+current+post1,
			                  'pred2+pred1'+pred2+pred1,
			                  'post1+post2'+post1+post2,
			                  'pred3+pred2'+pred3+pred2,
			                  'post2+post3'+post2+post3,
			                  'pred3+pred2+pred1'+pred3+pred2+pred1,
			                  'post1+post2+post3'+post1+post2+post3,
			                  'pred1+current+post1'+pred1+current+post1,
			                  'pred1+current'+pred1+current,
			                  'current+post1+post2'+current+post1+post2,
			                  ]
			yield feature_vector

	# edge features
	def generate_edge_features(self, Y_his, j, k):
		i = len(Y_his) 
		pred2 = tags[Y_his[i-1][j]] if i-1 >= 0 else 'start'
		pred1 = tags[j]
		current = tags[k]
		feature_vector = ['pred2+current'+pred2+current,
						  'pred1+current'+pred1+current,
						  #'pred2+pred1+current'+pred2+pred1+current,
						  ]
		return feature_vector

	def generate_edge_features_update(self, Y):
		for i in range(1, len(Y)):
			pred2 = Y[i-2] if i-2 >= 0 else 'start'
			pred1 = Y[i-1]
			current = Y[i]
			feature_vector = [
						  'pred2+current'+pred2+current,
						  'pred1+current'+pred1+current,
						  #'pred2+pred1+current'+pred2+pred1+current,
						  ]
			yield feature_vector


	def edge_weight(self, Y_his, j, k):
		feature_vector = self.generate_edge_features(Y_his, j, k)
		return sum(list(map(self.get_weight, feature_vector)))

	def viterbi(self, X):
		length = len(X)

		# node feature weight
		node_weights = [[sum(self.get_weight(tag+feature) for feature in features)
		              for tag in tags]
		              for features in self.generate_node_features(X)]
		node_weights[0][2] = node_weights[0][4] = node_weights[0][6] = -float('inf')     # first can't be I-
		

		# forward DP culculate the best pred for every node, alpha[i][j] max weight at index i with tag j
		alphas = [[(sum_weight_specific_tag, -1) for sum_weight_specific_tag in node_weights[0]]]
		Y_his = []
		for pos in range(length-1):
			# j: tag in index i
			# k: tag in index i+1
			alphas.append([max((alphas[pos][j][0]+self.edge_weight(Y_his, j, k)+node_weights[pos+1][k], j)   
			  		      for j in range(7)) for k in range(7)])    
			Y_his.append([alphas[pos+1][j][1] for j in range(7)])


		# backward find the best chain
		current = max([alphas[-1][j] for j in range(7)], key=lambda t: t[0])
		tag_chain = []
		for i in range(length):
			tag_chain.append(tags[current[1]])
			current = alphas[-i-1][current[1]]
		prediction = list(reversed(tag_chain))

		return prediction
		
	def beam_search(self, X, beam_width):
		pass

	def print_edge_weights(self):
		print()
		print('%-9s'%'',end='')
		for i in tags:
			print("%-8s"%i, end=' ')
		print()
		for i in tags:
			print("%-8s"%i, end=' ')
			for j in tags:
				print("%-8.0f"%self.get_weight('pred1+current'+i+j),end=' ')
			print()


def save_model(model, path):
	with open(path, 'wb') as f:
		pickle.dump(model, f)

def load_model(path):
	with open(path, 'rb') as f:
		model = pickle.load(f)
		return model

def main():
	stp = StructuredPerceptron()
	save_model(stp, model_dir+model_name)

if __name__ == "__main__":
	main()