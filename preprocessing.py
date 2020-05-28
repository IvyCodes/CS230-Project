import pandas as pd
import collections
import re
from sklearn.model_selection import train_test_split
import numpy as np


# names=['title', 'author', 'likes', 'comments', 'story', 'timestamp', 'verdict']
verdict_tags = ['YTA', 'ESH', 'NAH', 'NTA']

def remove_non_aita_entries(df):
	"""
	Removes entries with verdicts that are not NTA, NAH, ESH or YTA
	"""
	unk_idx = df[~df['verdict'].isin(verdict_tags)].index
	df.drop(unk_idx, inplace = True)

def drop_empty_stories(df):
	removed_idx = df[(df['story'] == '[removed]')].index
	df.drop(removed_idx, inplace = True)
	removed_idx = df[(df['story'] == '[deleted]')].index
	df.drop(removed_idx, inplace = True)
	removed_idx = df[(df['story'].isnull())].index
	df.drop(removed_idx, inplace = True)

def standardize_verdicts(df):
	"""
	Changes verdicts from text to NTA, NAH, ESH, YTA, or unknown
	"""
	verdicts = df.verdict
	verdcit_tags_verbose = ['Asshole', 'Everyone Sucks', 'No A-holes here', 'Not the A-hole']

	verbose_to_tag = {v_verbose: v for (v,v_verbose) in zip(verdict_tags, verdcit_tags_verbose)}
	# print(aita_dict)
	verdicts = [verbose_to_tag.get(v,'unk') for v in verdicts]
	df.verdict = verdicts

def add_verdict_num(df):
	"""
	Adds row verdict_num where verdict is cast to a numeric value
	"""
	tag_to_num = {v:n for (n,v) in enumerate(verdict_tags)}
	# print([tag_to_num.get(v, 0) for v in df.verdict])
	df['verdict_num'] = [tag_to_num.get(v, 0) for v in df.verdict]

def strip_text(text, keep_periods = False):
	text = text.lower()
	if keep_periods:
		text = re.sub(r'[^\w .]+', '', text)
	else:
		text = re.sub(r'[^\w ]+', '', text)
	return(text)


def strip_df_text(df):
	stories = [strip_text(story) for story in df.story]
	df.story = stories

	titles = [strip_text(title) for title in df.title]
	df.title = titles

def get_corpus_freq(df):
	'''
	Returns words from titles and stories and their frequencies
	'''
	# print(df.story)
	corpus = collections.defaultdict(int)
	for story in df.story:
		# print(story)
		story = story.split(" ")
		for word in story:
			corpus[word] += 1

	for title in df.title:
		title = title.split(" ")
		for word in title:
			corpus[word] += 1
	return corpus.items()

def get_corpus(df):
	'''
	Returns all sentences as a string
	'''
	# print(df.story)
	corpus = ""
	for story in df.story:
		corpus += strip_text(story, keep_periods = True) + " "

	for title in df.title:
		
		corpus += strip_text(title) + ". "
	return corpus

def get_word_to_idx(corpus, min_freq = 500):
	# words = [word for (word,freq) in corpus if freq >= min_freq]

	# corpus = list(corpus)
	# corpus.sort(key = lambda x: x[1])
	# print(corpus[0:int(0.999*len(corpus))])
	# words = [word for (word,freq) in corpus[0:int(0.999*len(corpus))] if freq >= min_freq]
	# return {word:i+1 for i,word in enumerate(words)} # add 1 to preserve space for unk -> 0
	words = ['deep', 'sons', 'slowly', 'insists', 'reasoning', 'originally', 'higher', 'pushed', 'obvious', 'discuss', 'rooms', 'handled', 'joined', 'hired', 'household', 'alright', 'risk', 'hurts', '22', 'ordered', 'fed', 'adults', 'pointed', 'speaking', 'neighborhood', 'aitaedit', 'perspective', 'dropped', 'damn', 'unfortunately', 'brings', 'band', 'wouldve', 'justified', 'ignoring', 'attending', 'informed', 'animals', 'trouble', 'wed', 'forget', 'fell', 'figured', 'shot', 'shocked', 'cousins', 'legal', 'pulled', 'cold', 'hates', 'cheating', 'instagram', 'pays', 'hope', 'jealous', 'defensive', 'owe', 'decent', 'terrible', 'blew', 'therefore', 'serious', 'choice', 'treated', 'travel', 'btw', 'daily', 'treat', 'sports', 'supposed', 'begin', 'form', 'visited', 'homework', 'social', 'mistake', 'ruin', 'cost', 'relationships', 'ignore', 'yard', 'across', 'force', 'complain', 'slept', 'drove', 'access', 'added', 'majority', 'honest', 'promised', 'reached', 'pm', 'stupid', 'studying', 'ta', 'generally', 'crush', 'opportunity', 'letter']
	words = words[0:min_freq]
	return {word:i+1 for i,word in enumerate(words)} # add 1 to preserve space for unk -> 0

def text_to_vec(text, word_to_idx):
	vec = [0] * (len(word_to_idx) + 1)
	for word in strip_text(text).split(" "):
		idx = word_to_idx.get(word, 0) # returns 0 if word not in vocabulary
		vec[idx] += 1

	return vec
	# return np.array(vec)/sum(vec)

def add_text_vec(df, word_to_idx = None, min_freq = None):
	if word_to_idx is None:
		word_to_idx = get_word_to_idx(get_corpus_freq(df), min_freq = min_freq)
	title_vecs = [text_to_vec(title, word_to_idx) for title in df.title]
	story_vecs = [text_to_vec(story, word_to_idx) for story in df.story]
	# print(title_vecs)
	# print(story_vecs)
	df['title_vec'] = title_vecs
	df['story_vec'] = story_vecs
	return word_to_idx

def get_data_with_vecs(filename, min_freq = 1000, word_to_idx = None):
	aita_df = pd.read_csv(filename)
	aita_minimal = aita_df[['title', 'story', 'verdict']]
	# aita_plus = aita_df[['title', 'author', 'likes', 'story', 'timestamp', 'verdict']]
	standardize_verdicts(aita_minimal)
	remove_non_aita_entries(aita_minimal)
	add_verdict_num(aita_minimal)
	corpus = list(get_corpus_freq(aita_minimal))
	strip_df_text(aita_minimal)
	word_to_idx = add_text_vec(aita_minimal, word_to_idx = word_to_idx, min_freq = min_freq)
	# print(word_to_idx)
	return(word_to_idx, aita_minimal)

def get_data(filename):
	aita_df = pd.read_csv(filename)
	aita_minimal = aita_df[['title', 'story', 'verdict']]

	return aita_minimal



# def main(min_freq = 1000):
# 	aita_df = pd.read_csv('data/aita.csv', names = ['id', 'title', 'score', 'num_comments', 'story', 'created', 'verdict'])
# 	aita_minimal = aita_df[['title', 'story', 'verdict']]
# 	drop_empty_stories(aita_minimal)
# 	standardize_verdicts(aita_minimal)
# 	remove_non_aita_entries(aita_minimal)


# 	train, test= train_test_split(aita_minimal, test_size=0.4)
# 	test, valid = train_test_split(test, test_size=0.5)

# 	train.to_csv('data/aita_minimal_train.csv', mode = 'w+', header = ['title', 'story', 'verdict'])
# 	valid.to_csv('data/aita_minimal_valid.csv', mode = 'w+', header = ['title', 'story', 'verdict'])
# 	test.to_csv('data/aita_minimal_test.csv', mode = 'w+', header = ['title', 'story', 'verdict'])

# main()
