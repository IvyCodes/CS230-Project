import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn
import util
import preprocessing


def run_with_min_freq(min_freq):
	word_to_idx, train = preprocessing.get_data_with_vecs('data/aita_minimal_train.csv', min_freq)
	# print(word_to_idx)
	_, test = preprocessing.get_data_with_vecs('data/aita_minimal_valid.csv', min_freq, word_to_idx)
	# print(train)

	X_train = util.get_X_vec(train)
	Y_train = util.get_Y_num(train)
	X_test = util.get_X_vec(test)
	Y_test = util.get_Y_num(test)

	# print(X_train)
	# print(X_train.shape)
	# num_words.append(X_train.shape[1])
	# print("X_train")

	linear_regressor = LinearRegression()
	linear_regressor.fit(X_train,Y_train)

	Y_pred_train = linear_regressor.predict(X_train)
	train_mse = sklearn.metrics.mean_squared_error(Y_train, Y_pred_train)
	# train_mses.append(train_mse)

	Y_pred_test = linear_regressor.predict(X_test)
	test_mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred_test)
	# test_mses.append(test_mse)

	Y_pred_baseline = [Y_train.mean()]*len(Y_test)
	baseline_mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred_baseline)
	# baseline_mses.append(baseline_mse)

	# print("Coefficients:")
	# coefs = list(enumerate(linear_regressor.coef_))
	# coefs.sort(key = lambda x: np.abs(x[1]))
	# top_50 = [x[0] for x in coefs[0:100]]
	# idx_to_word = {value:key for key,value in word_to_idx.items()}
	# print([idx_to_word[i] for i in top_50])
	return (X_train.shape[1], train_mse, test_mse, baseline_mse, word_to_idx, Y_pred_test, Y_test)


def find_min_freq():
	train_mses = []
	test_mses = []
	baseline_mses = []
	min_freqs = []
	num_words = []

	for i in range(0,20):
		# min_freq = i*2500+1000

		# for 0.999 of corpus
		# min_freq = i*5+1000

		# for top words
		min_freq = i+1

		num_words, train_mse, test_mse, baseline_mse, word_to_idx, _, _ = run_with_min_freq(min_freq)

		min_freqs.append(min_freq)

		train_mses.append(train_mse)

		test_mses.append(test_mse)

		baseline_mses.append(baseline_mse)

		
		# print(test_mse)
		# print(Y_pred_baseline)
	# print(train_mses)
	# print(test_mses)
	plt.plot(min_freqs, train_mses, label = 'train')
	plt.plot(min_freqs, test_mses, label = 'test')
	plt.plot(min_freqs, baseline_mses, label = 'baseline')
	plt.xlabel('minimum word frequency')
	plt.ylabel('MSE')
	plt.legend()
	plt.show()

	results = list(enumerate(test_mses))
	results.sort(key = lambda x: x[1])
	print(results)
	print(min_freqs)
	print(num_words)
	print(word_to_idx)

def run():
	num_words, train_mse, test_mse, baseline_mse, word_to_idx, Y_pred_test, Y_test = run_with_min_freq(1)

	#For using top_50 words
	# num_words, train_mse, test_mse, baseline_mse, word_to_idx, Y_pred_test, Y_test = run_with_min_freq(100)
	

	sums = {}
	counts = {}
	for y_pred, y in zip(Y_pred_test, Y_test):
		sums[y] = sums.get(y,0) + y_pred
		counts[y] = counts.get(y,0) + 1
	for key in sums:
		print(key)
		print(sums[key]/counts[key])
		print('')

	print(word_to_idx)
	print(test_mse)
	plt.scatter(Y_test, Y_pred_test, s = 1)
	plt.xlabel('Actual Rating')
	plt.ylabel('Predicted Rating')
	plt.ylim((-.1,3.1))
	plt.xlim((-.1,3.1))
	plt.show()

# run()
find_min_freq()