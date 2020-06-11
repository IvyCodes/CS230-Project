from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec
import preprocessing
import numpy as np
from sklearn.neural_network import MLPClassifier
import util
import sklearn
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils import resample
import pandas as pd
#####
# w2v-e with svm
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

	# print(data)
	model = Word2Vec(data, min_count = 1, size = 30, window = 5, sg = 1)
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

# # Balancing training set
# train_NTA = train[train.verdict == 'NTA']
# train_YTA = train[train.verdict == 'YTA']
# train_NAH = train[train.verdict == 'NAH']
# train_ESH = train[train.verdict == 'ESH']

# train_NTA = train_NTA.sample(int(1.*len(train_ESH)), random_state=0)
# train_YTA = train_YTA.sample(int(1.*len(train_ESH)), random_state=0)
# train_NAH = train_NAH.sample(int(1.*len(train_ESH)), random_state=0)

# train = pd.concat([train_NTA, train_YTA, train_NAH, train_ESH])
# train = train.sample(frac=1, random_state=0)


# # Changing labels
# train.verdict[train.verdict == 'NAH'] = 'YTA'
# train.verdict[train.verdict == 'ESH'] = 'YTA'
# test.verdict[test.verdict == 'NAH'] = 'YTA'
# test.verdict[test.verdict == 'ESH'] = 'YTA'
# # train[train.verdict == 'NAH']
# # train[train.verdict == 'ESH']

# train_NTA = train[train.verdict == 'NTA']
# train_YTA = train[train.verdict == 'YTA']
# # # train_NTA = train_NTA.sample(int(1.1*len(train_ESH)), random_state=0)
# train_NTA = train_NTA.sample(int(0.98*len(train_YTA)), random_state=0)
# # # train_NAH = train_NAH.sample(int(1.*len(train_ESH)), random_state=0)

# train = pd.concat([train_NTA, train_YTA])
# train = train.sample(frac=1, random_state=0)



# Changing labels with more correct splits
train.verdict[train.verdict == 'NAH'] = 'NTA'
train.verdict[train.verdict == 'ESH'] = 'YTA'
test.verdict[test.verdict == 'NAH'] = 'NTA'
test.verdict[test.verdict == 'ESH'] = 'YTA'
# train[train.verdict == 'NAH']
# train[train.verdict == 'ESH']

train_NTA = train[train.verdict == 'NTA']
train_YTA = train[train.verdict == 'YTA']
# train_NTA = train_NTA.sample(int(1.1*len(train_ESH)), random_state=0)
train_NTA = train_NTA.sample(int(.905*len(train_YTA)), random_state=0)
# train_NAH = train_NAH.sample(int(1.*len(train_ESH)), random_state=0)

train = pd.concat([train_NTA, train_YTA])
train = train.sample(frac=1, random_state=0)



# train_w2v_model(train)
model = get_w2v_model()
np.save('w2v.npy', model.wv.syn0)

add_vecs(model, train)
add_vecs(model, test)

Y_train = train.verdict
Y_test = test.verdict
# print(np.isnan(X_train))
print(Y_train.value_counts())
print(Y_test.value_counts())


X_train = util.get_X_vec(train)
X_test = util.get_X_vec(test)



linear_regressor = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50,30,50), random_state=0)
linear_regressor.fit(X_train,Y_train)


# train_mse = sklearn.metrics.mean_squared_error(convert_to_num(Y_train), convert_to_num(Y_pred_train))

# print(train_mse)

# plt.scatter(convert_to_num(Y_train), convert_to_num(Y_pred_train), s = 1)
# plt.xlabel('Actual Rating')
# plt.ylabel('Predicted Rating')
# plt.ylim((-.1,3.1))
# plt.xlim((-.1,3.1))
# plt.show()
Y_pred_test = linear_regressor.predict(X_test)
# print(Y_train)
Y_pred_test = pd.Index(Y_pred_test)
print(Y_pred_test.value_counts())

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
# class_names = ['NTA', '']
titles_options = [("P2V-DNN Test Set Confusion Matrix:\n(no Undersampling)", None)]
# titles_options = [("Reflex Validation Set Confusion Matrix:", None)]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(linear_regressor, X_test, Y_test,
                                 # display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize,
                                 values_format='d')
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()




Y_pred_train = linear_regressor.predict(X_train)
# print(Y_train)
Y_pred_train = pd.Index(Y_pred_train)
print(Y_pred_train.value_counts())
titles_options = [("P2V-DNN Training Set Confusion Matrix:\n(Undersampling with Unequal Proportions)", None)]
# titles_options = [("Reflex Training Set Confusion Matrix:", None)]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(linear_regressor, X_train, Y_train,
                                 # display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize,
                                 values_format='d')
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()

