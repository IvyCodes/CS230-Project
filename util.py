import pandas as pd
import re
import numpy as np

def string_to_list(text):
	text = re.sub(r'[^0-9,]+', '', text)
	text = text.split(',')

	to_list = []
	for s in text:
		to_list.append(int(s))
	return to_list

def get_X(df):
	return 0

def get_Y_num(df):
	return df.verdict_num

def get_X_vec(df):
	vecs = []
	for vec in df.story_vec:
		vecs.append(vec)
	# print(np.array(vecs))
	return np.array(vecs)