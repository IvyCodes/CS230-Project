from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec
import preprocessing
import numpy as np
from sklearn.linear_model import LogisticRegression
import util
import sklearn
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
# import pickle
# with open('pickle.dat', 'wb') as pickle_file:
# 	pickle.dump(model, pickle_file)

#####
# w2v-e with linear
#####

def train_w2v_model(train):
	'''
	trains model, only needs to be done once
	'''
	corpus = preprocessing.get_corpus(train)

	data = []
	# iterate through each sentence in the file 
	for i in sent_tokenize(corpus): 
	    temp = [] 
	      
	    # tokenize the sentence into words 
	    for j in word_tokenize(i): 
	        temp.append(j.lower()) 
	  
	    data.append(temp) 

	print(data)
	model = Word2Vec(data, min_count = 1, size = 60, window = 5)
	model.save("word2vec.model")

def get_w2v_model():
	return Word2Vec.load("word2vec.model")

def get_words(text):
	text = text.replace("\n", " ")
	return word_tokenize(preprocessing.strip_text(text))

def add_vecs(w2v, df):
	title_vecs = []
	dim = w2v.vector_size
	title_vec = np.zeros(dim)
	for title in df.title:
		unks = 0
		words = get_words(title)
		for word in words:
			if (word in w2v.wv):
				title_vec += w2v.wv[word]
			else:
				unks +=1
		# print(title)
		title_vecs.append(title_vec/(len(words)-unks))
	# print(len(title_vecs))
	# print(df.shape)
	df['title_vec'] = title_vecs


	story_vecs = []
	dim = w2v.vector_size
	story_vec = np.zeros(dim)
	for story in df.story:
		unks = 0
		# print(story)
		words = get_words(story)
		for word in words:
			if(word in w2v.wv):
				story_vec += w2v.wv[word]
			else:
				unks +=1 
		if((len(words)-unks) == 0):
			print( story)
			print(words)
			print(unks)
			story_vecs.append(np.zeros(dim))
		else:
			story_vecs.append(story_vec/(len(words)-unks))
	print(story_vec/(len(words)-unks))
	df['story_vec'] = story_vecs

def convert_to_num(y):
	aita_dict = {'NTA':3, 'YTA':0, 'NAH':2, 'ESH':1}
	return [aita_dict[i] for i in y]


train = preprocessing.get_data('data/aita_minimal_train.csv')
test = preprocessing.get_data('data/aita_minimal_test.csv')
print(test.shape)
print(train.shape)

train_w2v_model(train)
model = get_w2v_model()

add_vecs(model, train)
add_vecs(model, test)

X_train = util.get_X_vec(train)
Y_train = train.verdict
X_test = util.get_X_vec(test)
Y_test = test.verdict
# print(np.isnan(X_train))
# print(X_train)

linear_regressor = LogisticRegression(max_iter = 10000)
linear_regressor.fit(X_train,Y_train)

Y_pred_train = linear_regressor.predict(X_train)
print(Y_pred_train)
train_mse = sklearn.metrics.mean_squared_error(convert_to_num(Y_train), convert_to_num(Y_pred_train))

print(train_mse)

# plt.scatter(convert_to_num(Y_train), convert_to_num(Y_pred_train), s = 1)
# plt.xlabel('Actual Rating')
# plt.ylabel('Predicted Rating')
# plt.ylim((-.1,3.1))
# plt.xlim((-.1,3.1))
# plt.show()

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
# class_names = ['NTA', '']
titles_options = [("Confusion matrix", None)]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(linear_regressor, X_test, Y_test,
                                 # display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()

