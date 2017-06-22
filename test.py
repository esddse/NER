# coding:utf-8

from structured_perceptron import *
from util import *

import os
import pickle


test_model_path = './model/model_8'
output_path = './data/output'

def test():
	Xs, total_size = read_test_data()
	stp = load_model(test_model_path)

	predictions = []
	# tagging
	i = 1
	for X in Xs:
		print('sentence ',i)
		i += 1
		prediction = stp.viterbi(X)
		predictions.append(prediction)

	# output
	write_data(Xs, predictions, output_path)


def main():
	test()


if __name__ == "__main__":
	main()