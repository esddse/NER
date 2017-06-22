# coding:utf-8

from structured_perceptron import *
from util import *

import math
import matplotlib.pyplot as plt
import os

SET_NUM = 10
BATCH_SIZE = 128
PRINT_PER_STEP = 100
VALID_PER_STEP = 10000

validation_path_gold = './data/validation_gold'
validation_path_pred = './data/validation_pred'

# split data set into small set
def gen_cross_validation_set(data, size, set_num):
	
	set_size = math.ceil(size/set_num)
	data = random_shuffle(data, size)

	data_split = []
	for i in range(set_num):
		data_split.append(data[i*set_size:min((i+1)*set_size, size)])
	return data_split, set_size

def evalue(Y, prediction):
	length = len(Y)
	TP, FP, TN, FN = 0, 0, 0, 0
	for i in range(length):
		# positive
		if prediction[i] != 'O':
			# TP
			if Y[i] == prediction[i]:
				TP += 1
			# FP
			else:
				FP += 1
		# negative
		else:
			# TN
			if Y[i] == prediction[i]:
				TN += 1
			# FN
			else:
				FN += 1

	return length, TP, FP, TN, FN

def generate_train(Xs_train, Ys_train):
	length = len(Ys_train)
	datas = list(zip(Xs_train, Ys_train))
	for i in range(VALID_PER_STEP):
		index = np.random.randint(0,length)
		yield datas[index]
		
def train(total_epoch):
	
	# read training data
	Xs, Ys, total_size = read_train_data()
	data_sets, set_size = gen_cross_validation_set(list(zip(Xs, Ys)), total_size, SET_NUM)
	
	## cross validation
	for i in range(SET_NUM):
		# data preparation
		# ===========================================================
		
		# generate train set and dev set
		if i == 0:
			dev_set = data_sets[i]
			train_set = list_merge(data_sets[i+1:])
		elif i == SET_NUM-1:
			dev_set = data_sets[i]
			train_set = list_merge(data_sets[:i])
		else:
			dev_set = data_sets[i]
			train_set = list_merge(data_sets[:i]+data_sets[i+1:])

		Xs_train, Ys_train = zip(*train_set)
		Xs_dev, Ys_dev = zip(*dev_set)

		train_size = len(Ys_train)
		dev_size = len(Ys_dev)

		print('training set size: ',train_size, ', validation set size: ',dev_size)

		# train 
		# ===========================================================
		
		# init perceptron
		stp = StructuredPerceptron()
		# for plot
		F1s = []
		epochs = []

		total, TP, FP, TN, FN = 0, 0, 0, 0, 0
		for epoch in range(total_epoch):
			print('training epoch ', epoch, ' ....')
			step = 0
			
			# random shuffle
			#shuffled_data = random_shuffle(list(zip(Xs_train, Ys_train)), train_size)
			#Xs_train, Ys_train = zip(*shuffled_data)
		
			total, TP, FP, TN, FN = 0, 0, 0, 0, 0
			for X, Y in generate_train(Xs_train, Ys_train):
				prediction = stp.viterbi(X)
				stp.step += 1
				if prediction != Y:
					stp.update(X, Y, 1)
					stp.update(X, prediction, -1)
				delta_total, delta_TP, delta_FP, delta_TN, delta_FN = evalue(Y, prediction)
				total += delta_total
				TP += delta_TP
				FP += delta_FP
				TN += delta_TN
				FN += delta_FN
				if stp.step % PRINT_PER_STEP == 0:
					print ('precision = ', TP/(TP+FP+1e-30), '  recall = ', TP/(TP+FN+1e-30),'   step = ', step, '   epoch = ', epoch+1)
					step += 100
				#if stp.step % VALID_PER_STEP == 0:
				#	break
			

			# validate
			total, TP, FP, TN, FN = 0, 0, 0, 0, 0
			predictions = []
			for X, Y in zip(Xs_dev, Ys_dev):
				prediction = stp.viterbi(X)
				predictions.append(prediction)
				delta_total, delta_TP, delta_FP, delta_TN, delta_FN = evalue(Y, prediction)
				total += delta_total
				TP += delta_TP
				FP += delta_FP
				TN += delta_TN
				FN += delta_FN
			print ('precision = ', TP/(TP+FP+1e-30), '  recall = ', TP/(TP+FN+1e-30),'   step = validation', '   epoch = ', epoch+1)
			write_data(Xs_dev, Ys_dev, validation_path_gold)
			write_data(Xs_dev, predictions, validation_path_pred)
			result = os.popen('python2 evaluate.py '+validation_path_pred+' '+validation_path_gold).read()
			F1s.append(float(result.strip().split('\n')[-1].split(' ')[-1]))
			epochs.append(epoch+1)
			print(result)
			#stp.print_edge_weights()

		# average perceptron
		stp.average()
		# final validate
		total, TP, FP, TN, FN = 0, 0, 0, 0, 0
		predictions = []
		for X, Y in zip(Xs_dev, Ys_dev):
			prediction = stp.viterbi(X)
			predictions.append(prediction)
			delta_total, delta_TP, delta_FP, delta_TN, delta_FN = evalue(Y, prediction)
			total += delta_total
			TP += delta_TP
			FP += delta_FP
			TN += delta_TN
			FN += delta_FN
		print ('precision = ', TP/(TP+FP+1e-30), '  recall = ', TP/(TP+FN+1e-30),'   step = validation', '   epoch = ', epoch+1)
		write_data(Xs_dev, Ys_dev, validation_path_gold)
		write_data(Xs_dev, predictions, validation_path_pred)
		result = os.popen('python2 evaluate.py '+validation_path_pred+' '+validation_path_gold).read()
		print(result)
		F1s.append(float(result.strip().split('\n')[-1].split(' ')[-1]))
		epochs.append(epoch+2)

		# plot
		plt.plot(epochs, F1s)
		plt.savefig(model_dir+'figs/'+model_name+'training.jpg')
		


		# save model
		save_model(stp, model_dir+model_name+str(i))


def main():
	train(30)

if __name__ == "__main__":
	main()