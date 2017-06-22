# coding:utf-8

import numpy as np

train_data_path = "./data/ner_trn"
test_data_path = "./data/ner_tst_wol"


def list_merge(lsts):
	merged_lst = []
	for lst in lsts:
		merged_lst.extend(lst)
	return merged_lst

def random_shuffle(data, size):
	data = np.array(data)
	shuffled_indices = np.random.permutation(np.arange(size))
	shuffled_data = data[shuffled_indices]
	
	return shuffled_data

def write_data(Xs, Ys, path):
	with open(path, 'w', encoding='utf8') as f:
		for X,Y in zip(Xs, Ys):
			for x, y in zip(X, Y):
				f.write(x+' '+y+'\n')
			f.write('\n')

# read the training data
def read_train_data():
	Xs, Ys = [], []
	with open(train_data_path, 'r', encoding='utf8') as f:
		sentences = f.read().strip().split('\n\n')
		for sentence in sentences:
			X, Y = [], []
			char_with_tags = sentence.strip().split('\n')
			#print(char_with_tags)
			for char_with_tag in char_with_tags:
				char_with_tag = char_with_tag.split(' ')
				X.append(char_with_tag[0])
				Y.append(char_with_tag[1])
			Xs.append(X)
			Ys.append(Y)

	data_size = len(Ys)
	print('load ', data_size, ' sentences!')
	return Xs, Ys, data_size

# read the test data
def read_test_data():
	Xs = []
	with open(test_data_path, 'r', encoding='utf8') as f:
		sentences = f.read().strip().split('\n\n')
		for sentence in sentences:
			Xs.append(sentence.strip().split('\n'))
	data_size = len(Xs)
	print('load', data_size, 'sentences!')
	return Xs, data_size	


def write_data(Xs, Ys, path):
	print('writing data to ',path)
	with open(path, 'w', encoding='utf8') as f:
		for X, Y in zip(Xs, Ys):
			for char, tag in zip(X, Y):
				f.write(char+' '+tag+'\n')
			f.write('\n')

def main():
	_, Ys = read_train_data()


if __name__ == "__main__":
	main()