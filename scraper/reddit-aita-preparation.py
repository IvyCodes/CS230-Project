#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from collections import defaultdict
import statistics
import _pickle as pickle
import json
import csv
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import collections
from datetime import datetime
plt.style.use('seaborn')
sns.set(style="whitegrid")


# In[2]:


aita_df = pd.read_csv("final_aita.csv")
print("The size of the dataframe is {}.".format(aita_df.shape))
aita_df.head()


# In[3]:


aita_df.columns


# # Missing Data
# 
# The missing data isn't because we have incorrectly scrapped the data from the subreddit but rather it is when a moderator(mod) choses to lock the thread early. When a mod does this, it stops all new comments from being uploaded. So the missing data is from user comments where it was locked after the 2nd or 3rd user comment thus the missing data. Therefore we will just fill these missing data with 'No comments allowed'.

# In[4]:


aita_df.isnull().sum()


# In[5]:


aita_df[aita_df['body'].isnull()]


# In[6]:


aita_df['body'].fillna('No story. META post', inplace = True)


# In[7]:


aita_df[aita_df['flair'].isnull()]


# In[8]:


aita_df.loc[(aita_df['id'] == 'cr4uat'), 'flair'] = 'Asshole'


# In[9]:


aita_df['flair'].fillna('Not the A-hole', inplace=True)


# In[10]:


aita_df[aita_df.isnull().any(axis=1)]


# In[11]:


aita_df.fillna("No comments allowed", inplace=True)


# # Flair 

# In[12]:


aita_df['flair'].value_counts()


# In[13]:


aita_df['flair'].replace(['Not the A-hole', 'not the a-hole','Not the A-hole (oof)', 
                          'Asshole','Record Setting Asshole','weeabo h8r', 'Shitpost', 'META Asshole',
                          'No A-holes here', 
                          'Everyone Sucks', 'everyone sucks', 
                          'Update', 
                          'Not enough info', 'TL;DR',
                          'Probably Fake', 'probably fake', 'Fake Story',
                          'Actually Meta'], 
                         ['NTA', 'NTA', 'NTA', 
                          'YTA', 'YTA', 'YTA', 'YTA', 'YTA',
                          'NAH', 
                          'ESH', 'ESH', 
                          'UPDATE', 
                          'INFO', 'INFO',
                          'FAKE', 'FAKE', 'FAKE',
                          'META'], inplace = True)


# In[14]:


aita_df['flair'].value_counts()


# In[15]:


squarify.plot(sizes=aita_df.flair.value_counts(), label=aita_df.flair.value_counts().index, alpha =.7, 
              color=['#394a6d', '#3c9d9b','#4baea0', '#b6e6bd', '#f1f0cf', '#f0c9c9', '#f2a6a6', '#f8a978'])
plt.axis('off')
plt.show()


# In[16]:


plt.figure(figsize=(20,10))
sns.barplot(x=aita_df.flair.value_counts().index, y=aita_df.flair.value_counts(), color='pink')


# # Created On - Feature

# In[17]:


aita_df['created'] = aita_df['created'].apply(lambda x: datetime.utcfromtimestamp(x))


# In[18]:


aita_df['year_month'] = aita_df['created'].map(lambda x: x.strftime('%Y-%m'))


# In[19]:


aita_df['year'] = aita_df['created'].apply(lambda x: x.year)


# In[20]:


aita_df['month'] = aita_df['created'].apply(lambda x: x.month)


# In[21]:


aita_df['year'].value_counts()


# In[22]:


aita_df.columns


# In[23]:


year_month_count = aita_df['year_month'].value_counts().sort_index()
plt.figure(figsize=(20, 10))
sns.barplot(x=year_month_count.sort_index().index, y=year_month_count, color='lightgreen')


# In[24]:


year_month_comments = aita_df[['year_month', 'num_comments']].groupby(['year_month']).sum()
plt.figure(figsize=(20, 10))
sns.barplot(x=year_month_comments.index, y=year_month_comments.num_comments, color='lightgreen')


# In[25]:


year_month_score = aita_df[['year_month', 'score']].groupby(['year_month']).mean()
plt.figure(figsize=(20, 10))
sns.barplot(x=year_month_score.index, y=year_month_score.score, color='lightgreen')


# # Word Count - Feature
# 
# A key feature that we can extract from the text is the word count. The word count will include stopwords but it will give us insights like how much the average user types for their title, story and comments. And we can see if these word counts are affected by seasonality or certain themes later on.

# In[26]:


aita_df['body_wordcount'] = aita_df['body'].apply(lambda x: len(x.split()))
aita_df['title_wordcount'] = aita_df['title'].apply(lambda x: len(x.split()))


# In[27]:


uc = []
for num in range(1,11):
    uc.append('uc{}_wordcount'.format(num))
counter = 0
for col in range(1, 11):
    aita_df['{}'.format(uc[counter])] = aita_df['uc{}'.format(col)].apply(lambda x: len(x.split()))
    counter += 1


# In[28]:


year_month_bodywc = aita_df[['year_month', 'body_wordcount']].groupby(['year_month']).sum()
plt.figure(figsize=(20, 10))
sns.barplot(x=year_month_bodywc.index, y=year_month_bodywc.body_wordcount, color='teal')


# In[29]:


year_month_titlewc = aita_df[['year_month', 'title_wordcount']].groupby(['year_month']).sum()
plt.figure(figsize=(20, 10))
sns.barplot(x=year_month_titlewc.index, y=year_month_titlewc.title_wordcount, color='teal')


# In[30]:


year_month_uc1wc = aita_df[['year_month', 'uc1_wordcount']].groupby(['year_month']).sum()
plt.figure(figsize=(20, 10))
sns.barplot(x=year_month_uc1wc.index, y=year_month_uc1wc.uc1_wordcount, color='teal')


# In[31]:


year_month_uc2wc = aita_df[['year_month', 'uc2_wordcount']].groupby(['year_month']).sum()
plt.figure(figsize=(20, 10))
sns.barplot(x=year_month_uc2wc.index, y=year_month_uc2wc.uc2_wordcount, color='teal')


# In[32]:


year_month_uc3wc = aita_df[['year_month', 'uc3_wordcount']].groupby(['year_month']).sum()
plt.figure(figsize=(20, 10))
sns.barplot(x=year_month_uc3wc.index, y=year_month_uc3wc.uc3_wordcount, color='teal')


# In[33]:


year_month_uc4wc = aita_df[['year_month', 'uc4_wordcount']].groupby(['year_month']).sum()
plt.figure(figsize=(20, 10))
sns.barplot(x=year_month_uc4wc.index, y=year_month_uc4wc.uc4_wordcount, color='teal')


# In[34]:


year_month_bodywc


# In[35]:


year_month_uc5wc = aita_df[['year_month', 'uc5_wordcount']].groupby(['year_month']).sum()
plt.figure(figsize=(20, 10))
sns.barplot(x=year_month_uc5wc.index, y=year_month_uc5wc.uc5_wordcount, color='teal')


# In[36]:


year_month_uc6wc = aita_df[['year_month', 'uc6_wordcount']].groupby(['year_month']).sum()
plt.figure(figsize=(20, 10))
sns.barplot(x=year_month_uc6wc.index, y=year_month_uc6wc.uc6_wordcount, color='teal')


# In[37]:


year_month_uc7wc = aita_df[['year_month', 'uc7_wordcount']].groupby(['year_month']).sum()
plt.figure(figsize=(20, 10))
sns.barplot(x=year_month_uc7wc.index, y=year_month_uc7wc.uc7_wordcount, color='teal')


# In[38]:


year_month_uc8wc = aita_df[['year_month', 'uc8_wordcount']].groupby(['year_month']).sum()
plt.figure(figsize=(20, 10))
sns.barplot(x=year_month_uc8wc.index, y=year_month_uc8wc.uc8_wordcount, color='teal')


# In[39]:


year_month_uc9wc = aita_df[['year_month', 'uc9_wordcount']].groupby(['year_month']).sum()
plt.figure(figsize=(20, 10))
sns.barplot(x=year_month_uc9wc.index, y=year_month_uc9wc.uc9_wordcount, color='teal')


# In[40]:


year_month_uc10wc = aita_df[['year_month', 'uc10_wordcount']].groupby(['year_month']).sum()
plt.figure(figsize=(20, 10))
sns.barplot(x=year_month_uc10wc.index, y=year_month_uc10wc.uc10_wordcount, color='teal')


# # Age - feature
# 
# Another key insight we can obtain from some of the stories is that the user sometimes place age. By using the age we can get a general understanding of the common age bracket. For now we will just grab the first age of male and female in the stories. Another option is using the median of all ages located. 

# In[41]:


male_age_re = re.compile('\d\d[Mm]')
female_age_re = re.compile('\d\dF|m')
male_age = re.findall(male_age_re, aita_df['body'][3])
female_age = re.findall(female_age_re, aita_df['body'][3])


# In[42]:


male_age_re = re.compile('\d\d[Mm]')
female_age_re = re.compile('\d\d[Ff]')
aita_df['male_age'] = aita_df['body'].apply(lambda x: re.findall(male_age_re, x))
aita_df['male_age'] = aita_df['male_age'].apply(lambda x: 0 if len(x)== 0 else x[0])
aita_df['female_age'] = aita_df['body'].apply(lambda x: re.findall(female_age_re, x))
aita_df['female_age'] = aita_df['female_age'].apply(lambda x: 0 if len(x)== 0 else x[0])
aita_df['male_age'] = aita_df['male_age'].apply(lambda x: re.sub(r'M|m', r'', str(x)))
aita_df['male_age'] = aita_df['male_age'].astype(np.int64)
aita_df['female_age'] = aita_df['female_age'].apply(lambda x: re.sub(r'F|f', r'', str(x)))
aita_df['female_age'] = aita_df['female_age'].astype(np.int64)


# # Word cleaning
# 
# We will begin cleaning the text so we can prepare it for further analysis and modelling. We will be removing all digits, making all text lowercase and getting rid of punctuation and stopwords. Stopwords are words like 'all', 'same' 'if' which are words search engine use to ignore. Lemmatization is the process of reducing inflectional forms and derive forms of words related to a comman base. E.g. different -> differ, boy's -> boy.

# In[43]:


def word_digit_clean(word):
    word = ''.join([i for i in word if not i.isdigit()])    
    html_tags = re.compile('<.*?>')
    return word


# In[44]:


aita_df['body_clean'] = aita_df['body'].apply(lambda x: word_digit_clean(x))
aita_df['title_clean'] = aita_df['title'].apply(lambda x: word_digit_clean(x))
uc = []
for num in range(1,11):
    uc.append('uc{}_clean'.format(num))
counter = 0
for col in range(1, 11):
    aita_df['{}'.format(uc[counter])] = aita_df['uc{}'.format(col)].apply(lambda x: word_digit_clean(x))
    counter += 1


# In[45]:


tokenizer = RegexpTokenizer(r'\w+')
aita_df['body_clean'] = aita_df['body_clean'].apply(lambda x: tokenizer.tokenize(x.lower()))
aita_df['title_clean'] = aita_df['title_clean'].apply(lambda x: tokenizer.tokenize(x.lower()))
for col in range(1, 11):
    aita_df['uc{}_clean'.format(col)] = aita_df['uc{}_clean'.format(col)].apply(lambda x:tokenizer.tokenize(x.lower()))
    counter += 1


# In[46]:


def remove_stopwords(text):
    words = [word for word in text if word not in stopwords.words('english')]
    return words


# In[47]:


aita_df['body_clean'] = aita_df['body_clean'].apply(lambda x: remove_stopwords(x))
aita_df['title_clean'] = aita_df['title_clean'].apply(lambda x: remove_stopwords(x))
for col in range(1, 11):
    aita_df['uc{}_clean'.format(col)] = aita_df['uc{}_clean'.format(col)].apply(lambda x: remove_stopwords(x))
    counter += 1


# In[48]:


lemmatizer = WordNetLemmatizer()
def word_lemmatizer(text):
    lem_text = " ".join([lemmatizer.lemmatize(i) for i in text])
    return lem_text


# In[49]:


aita_df['body_clean'] = aita_df['body_clean'].apply(lambda x: word_lemmatizer(x))
aita_df['title_clean'] = aita_df['title_clean'].apply(lambda x: word_lemmatizer(x))
for col in range(1, 11):
    aita_df['uc{}_clean'.format(col)] = aita_df['uc{}_clean'.format(col)].apply(lambda x: word_lemmatizer(x))
    counter += 1


# # Collocation

# In[50]:


def colloc(word_pair):
    colloc_temp = dict()
    for colloc in word_pair:
        if colloc in colloc_temp:
            colloc_temp[colloc] += 1
        else:
            colloc_temp[colloc] = 1
    colloc = [[keys,values] for keys, values in colloc_temp.items() ]
    return colloc


# In[51]:


def colloc_tolist(word_pair):
    colloc_temp = []
    for row in word_pair:
        for col in row:
            colloc_temp.append(col)
    return colloc_temp


# In[52]:


def colloc_todict(word_pair):
    colloc_temp = dict()
    for colloc in word_pair:
        for row in colloc:
            if row in colloc_temp:
                colloc_temp[row] += 1
            else:
                colloc_temp[row] = 1
    return colloc_temp


# In[53]:


bg_measures = BigramAssocMeasures()
def dataframe_colloc(dataframe, column, target):
    dataframe[column] = dataframe[target].apply(lambda x: nltk.word_tokenize(x))
    dataframe[column] = dataframe[column].apply(lambda x: BigramCollocationFinder.from_words(x))
    dataframe[column] = dataframe[column].apply(lambda x: x.nbest(bg_measures.likelihood_ratio, 10))
    dataframe[column] = dataframe[column].apply(lambda x: colloc(x))
    return dataframe[column]


# In[54]:


aita_df['body_colloc'] = dataframe_colloc(aita_df, 'body_colloc', 'body_clean')
aita_df['title_colloc'] = dataframe_colloc(aita_df, 'title_colloc', 'title_clean')
aita_df["uc1_colloc"] = dataframe_colloc(aita_df, "uc1", "uc1_clean")
aita_df["uc2_colloc"] = dataframe_colloc(aita_df, "uc2", "uc2_clean")
aita_df["uc3_colloc"] = dataframe_colloc(aita_df, "uc3", "uc3_clean")
aita_df["uc4_colloc"] = dataframe_colloc(aita_df, "uc4", "uc4_clean")
aita_df["uc5_colloc"] = dataframe_colloc(aita_df, "uc", "uc5_clean")
aita_df["uc6_colloc"] = dataframe_colloc(aita_df, "uc", "uc6_clean")
aita_df["uc7_colloc"] = dataframe_colloc(aita_df, "uc", "uc7_clean")
aita_df["uc8_colloc"] = dataframe_colloc(aita_df, "uc", "uc8_clean")
aita_df["uc9_colloc"] = dataframe_colloc(aita_df, "uc", "uc9_clean")
aita_df["uc10_colloc"] = dataframe_colloc(aita_df, "uc", "uc10_clean")


# In[55]:


body_colloc = colloc_tolist(aita_df['body_colloc'])
body_colloc = colloc_todict(body_colloc)
body_od_colloc = collections.OrderedDict(sorted(body_colloc.items(), key = lambda x : x[1], reverse=True))
title_colloc = colloc_tolist(aita_df['title_colloc'])
title_colloc = colloc_todict(title_colloc)
title_od_colloc = collections.OrderedDict(sorted(title_colloc.items(), key = lambda x : x[1], reverse=True))
uc1_colloc = colloc_tolist(aita_df["uc1_colloc"])
uc1_colloc = colloc_todict(uc1_colloc)
uc1_od_colloc = collections.OrderedDict(sorted(uc1_colloc.items(), key = lambda x : x[1], reverse=True))
uc2_colloc = colloc_tolist(aita_df["uc2_colloc"])
uc2_colloc = colloc_todict(uc2_colloc)
uc2_od_colloc = collections.OrderedDict(sorted(uc2_colloc.items(), key = lambda x : x[1], reverse=True))
uc3_colloc = colloc_tolist(aita_df["uc3_colloc"])
uc3_colloc = colloc_todict(uc3_colloc)
uc3_od_colloc = collections.OrderedDict(sorted(uc3_colloc.items(), key = lambda x : x[1], reverse=True))
uc4_colloc = colloc_tolist(aita_df["uc4_colloc"])
uc4_colloc = colloc_todict(uc4_colloc)
uc4_od_colloc = collections.OrderedDict(sorted(uc4_colloc.items(), key = lambda x : x[1], reverse=True))
uc5_colloc = colloc_tolist(aita_df["uc5_colloc"])
uc5_colloc = colloc_todict(uc5_colloc)
uc5_od_colloc = collections.OrderedDict(sorted(uc5_colloc.items(), key = lambda x : x[1], reverse=True))
uc6_colloc = colloc_tolist(aita_df["uc6_colloc"])
uc6_colloc = colloc_todict(uc6_colloc)
uc6_od_colloc = collections.OrderedDict(sorted(uc6_colloc.items(), key = lambda x : x[1], reverse=True))
uc7_colloc = colloc_tolist(aita_df["uc7_colloc"])
uc7_colloc = colloc_todict(uc7_colloc)
uc7_od_colloc = collections.OrderedDict(sorted(uc7_colloc.items(), key = lambda x : x[1], reverse=True))
uc8_colloc = colloc_tolist(aita_df["uc8_colloc"])
uc8_colloc = colloc_todict(uc8_colloc)
uc8_od_colloc = collections.OrderedDict(sorted(uc8_colloc.items(), key = lambda x : x[1], reverse=True))
uc9_colloc = colloc_tolist(aita_df["uc9_colloc"])
uc9_colloc = colloc_todict(uc9_colloc)
uc9_od_colloc = collections.OrderedDict(sorted(uc9_colloc.items(), key = lambda x : x[1], reverse=True))
uc10_colloc = colloc_tolist(aita_df["uc10_colloc"])
uc10_colloc = colloc_todict(uc10_colloc)
uc10_od_colloc = collections.OrderedDict(sorted(uc10_colloc.items(), key = lambda x : x[1], reverse=True))


# In[56]:


colloc_df = pd.DataFrame()
colloc_df['body_pairs'] = list(body_od_colloc.keys())[:1000]
colloc_df['body_frequency'] = list(body_od_colloc.values())[:1000]
colloc_df['title_pairs'] = list(title_od_colloc.keys())[:1000]
colloc_df['title_frequency'] = list(title_od_colloc.values())[:1000]
colloc_df["uc1_pairs"] = list(uc1_od_colloc.keys())[:1000]
colloc_df["uc1_frequency"] = list(uc1_od_colloc.values())[:1000]
colloc_df["uc2_pairs"] = list(uc2_od_colloc.keys())[:1000]
colloc_df["uc2_frequency"] = list(uc2_od_colloc.values())[:1000]
colloc_df["uc3_pairs"] = list(uc3_od_colloc.keys())[:1000]
colloc_df["uc3_frequency"] = list(uc3_od_colloc.values())[:1000]
colloc_df["uc4_pairs"] = list(uc4_od_colloc.keys())[:1000]
colloc_df["uc4_frequency"] = list(uc4_od_colloc.values())[:1000]
colloc_df["uc5_pairs"] = list(uc5_od_colloc.keys())[:1000]
colloc_df["uc5_frequency"] = list(uc5_od_colloc.values())[:1000]
colloc_df["uc6_pairs"] = list(uc6_od_colloc.keys())[:1000]
colloc_df["uc6_frequency"] = list(uc6_od_colloc.values())[:1000]
colloc_df["uc7_pairs"] = list(uc7_od_colloc.keys())[:1000]
colloc_df["uc7_frequency"] = list(uc7_od_colloc.values())[:1000]
colloc_df["uc8_pairs"] = list(uc8_od_colloc.keys())[:1000]
colloc_df["uc8_frequency"] = list(uc8_od_colloc.values())[:1000]
colloc_df["uc9_pairs"] = list(uc9_od_colloc.keys())[:1000]
colloc_df["uc9_frequency"] = list(uc9_od_colloc.values())[:1000]
colloc_df["uc10_pairs"] = list(uc10_od_colloc.keys())[:1000]
colloc_df["uc10_frequency"] = list(uc10_od_colloc.values())[:1000]


# In[57]:


colloc_df


# In[58]:


colloc_df.drop(0, inplace = True)


# In[59]:


colloc_df.to_csv('colloc.csv')


# # Colloc Graphs

# In[60]:


sns.set(style="whitegrid")
plt.figure(figsize=(20, 10))
body_colloc_sns_plot = sns.barplot(x="body_frequency", y="body_pairs", data=colloc_df[:50], color = "salmon")


# In[61]:


sns.set(style="whitegrid")
plt.figure(figsize=(20, 10))
title_colloc_sns_plot = sns.barplot(x="title_frequency", y="title_pairs", data=colloc_df[:50], color = "salmon")


# In[62]:


sns.set(style="whitegrid")
plt.figure(figsize=(20, 10))
uc1_colloc_sns_plot = sns.barplot(x="uc1_frequency", y="uc1_pairs", data=colloc_df[:50], color = "salmon")


# In[63]:


sns.set(style="whitegrid")
plt.figure(figsize=(20, 10))
uc2_colloc_sns_plot = sns.barplot(x="uc2_frequency", y="uc2_pairs", data=colloc_df[:50], color = "salmon")


# In[64]:


sns.set(style="whitegrid")
plt.figure(figsize=(20, 10))
uc3_colloc_sns_plot = sns.barplot(x="uc3_frequency", y="uc3_pairs", data=colloc_df[:50], color = "salmon")


# In[65]:


sns.set(style="whitegrid")
plt.figure(figsize=(20, 10))
uc4_colloc_sns_plot = sns.barplot(x="uc4_frequency", y="uc4_pairs", data=colloc_df[:50], color = "salmon")


# In[66]:


sns.set(style="whitegrid")
plt.figure(figsize=(20, 10))
uc5_colloc_sns_plot = sns.barplot(x="uc5_frequency", y="uc5_pairs", data=colloc_df[:50], color = "salmon")


# In[67]:


sns.set(style="whitegrid")
plt.figure(figsize=(20, 10))
uc6_colloc_sns_plot = sns.barplot(x="uc6_frequency", y="uc6_pairs", data=colloc_df[:50], color = "salmon")


# In[68]:


sns.set(style="whitegrid")
plt.figure(figsize=(20, 10))
uc7_colloc_sns_plot = sns.barplot(x="uc7_frequency", y="uc7_pairs", data=colloc_df[:50], color = "salmon")
sns.set(style="whitegrid")


# In[69]:


sns.set(style="whitegrid")
plt.figure(figsize=(20, 10))
uc8_colloc_sns_plot = sns.barplot(x="uc8_frequency", y="uc8_pairs", data=colloc_df[:50], color = "salmon")


# In[70]:


sns.set(style="whitegrid")
plt.figure(figsize=(20, 10))
uc9_colloc_sns_plot = sns.barplot(x="uc9_frequency", y="uc9_pairs", data=colloc_df[:50], color = "salmon")


# In[71]:


sns.set(style="whitegrid")
plt.figure(figsize=(20, 10))
uc10_colloc_sns_plot = sns.barplot(x="uc10_frequency", y="uc10_pairs", data=colloc_df[:50], color = "salmon")


# # Word value counts 
# 
# We will used the clean text from the columns, title, story, usercomments1-10 and determine its frequency of all the words. The goal of this is to create visualisation of what are the common words used for these features. We will visualise it in two ways through a wordcloud and a bar graph. Wordclouds are great for everyday viewers to get some insights from the visualisation however flaws such as longer words appear bigger creates it harder to truly understand the data. Thus the bar graph will implemented to supplement the word cloud. 
# 
# Interactive dashboards will also be planned.

# In[72]:


body_wordlist = aita_df['body_clean'].str.split(expand = True).stack().value_counts().astype(int)
title_wordlist = aita_df['title_clean'].str.split(expand = True).stack().value_counts().astype(int)
uc1_wordlist = aita_df['uc1_clean'].str.split(expand = True).stack().value_counts().astype(int)
uc2_wordlist = aita_df['uc2_clean'].str.split(expand = True).stack().value_counts().astype(int)
uc3_wordlist = aita_df['uc3_clean'].str.split(expand = True).stack().value_counts().astype(int)
uc4_wordlist = aita_df['uc4_clean'].str.split(expand = True).stack().value_counts().astype(int)
uc5_wordlist = aita_df['uc5_clean'].str.split(expand = True).stack().value_counts().astype(int)
uc6_wordlist = aita_df['uc6_clean'].str.split(expand = True).stack().value_counts().astype(int)
uc7_wordlist = aita_df['uc7_clean'].str.split(expand = True).stack().value_counts().astype(int)
uc8_wordlist = aita_df['uc8_clean'].str.split(expand = True).stack().value_counts().astype(int)
uc9_wordlist = aita_df['uc9_clean'].str.split(expand = True).stack().value_counts().astype(int)
uc10_wordlist = aita_df['uc10_clean'].str.split(expand = True).stack().value_counts().astype(int)


# In[73]:


body_wordlist_dict = dict(body_wordlist)
title_wordlist_dict = dict(title_wordlist)
uc1_dict = dict(uc1_wordlist)
uc2_dict = dict(uc2_wordlist)
uc3_dict = dict(uc3_wordlist)
uc4_dict = dict(uc4_wordlist)
uc5_dict = dict(uc5_wordlist)
uc6_dict = dict(uc6_wordlist)
uc7_dict = dict(uc7_wordlist)
uc8_dict = dict(uc8_wordlist)
uc9_dict = dict(uc9_wordlist)
uc10_dict = dict(uc10_wordlist)


# In[74]:


body, body_frequency = list(body_wordlist_dict.keys())[:1000], list(body_wordlist_dict.values())[:1000]
title, title_frequency = list(title_wordlist_dict.keys())[:1000], list(title_wordlist_dict.values())[:1000]
uc1, uc1_frequency = list(uc1_dict.keys())[:1000], list(uc1_dict.values())[:1000]
uc2, uc2_frequency = list(uc2_dict.keys())[:1000], list(uc2_dict.values())[:1000]
uc3, uc3_frequency = list(uc3_dict.keys())[:1000], list(uc3_dict.values())[:1000]
uc4, uc4_frequency = list(uc4_dict.keys())[:1000], list(uc4_dict.values())[:1000]
uc5, uc5_frequency = list(uc5_dict.keys())[:1000], list(uc5_dict.values())[:1000]
uc6, uc6_frequency = list(uc6_dict.keys())[:1000], list(uc6_dict.values())[:1000]
uc7, uc7_frequency = list(uc7_dict.keys())[:1000], list(uc7_dict.values())[:1000]
uc8, uc8_frequency = list(uc8_dict.keys())[:1000], list(uc8_dict.values())[:1000]
uc9, uc9_frequency = list(uc9_dict.keys())[:1000], list(uc9_dict.values())[:1000]
uc10, uc10_frequency = list(uc10_dict.keys())[:1000], list(uc10_dict.values())[:1000]


# In[75]:


word_df = pd.DataFrame()
word_df = word_df.assign(body=body, body_frequency=body_frequency, title=title, title_frequency=title_frequency,
                        uc1=uc1, uc1_frequency=uc1_frequency, uc2=uc2, uc2_frequency=uc2_frequency, 
                        uc3=uc3, uc3_frequency=uc3_frequency, uc4=uc4, uc4_frequency=uc4_frequency, 
                        uc5=uc5, uc5_frequency=uc5_frequency, uc6=uc6, uc6_frequency=uc6_frequency, 
                        uc7=uc7, uc7_frequency=uc7_frequency, uc8=uc8, uc8_frequency=uc8_frequency, 
                        uc9=uc9, uc9_frequency=uc9_frequency, uc10=uc10, uc10_frequency=uc10_frequency)


# In[76]:


word_df.to_csv('word_frequncy.csv')


# # Bar graph of word count by date

# In[77]:


aita_df.columns


# # wordcloud by frequency of words

# In[78]:


body_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(body_wordlist_dict)
title_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(title_wordlist_dict)
uc1_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(uc1_dict)
uc2_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(uc2_dict)
uc3_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(uc3_dict)
uc4_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(uc4_dict)
uc5_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(uc5_dict)
uc6_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(uc6_dict)
uc7_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(uc7_dict)
uc8_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(uc8_dict)
uc9_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(uc9_dict)
uc10_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(uc10_dict)


# In[79]:


plt.imshow(body_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[80]:


plt.imshow(title_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[81]:


plt.imshow(uc1_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[82]:


plt.imshow(uc2_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[83]:


plt.imshow(uc3_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[84]:


plt.imshow(uc4_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[85]:


plt.imshow(uc5_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[86]:


plt.imshow(uc6_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[87]:


plt.imshow(uc7_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[88]:


plt.imshow(uc8_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[89]:


plt.imshow(uc9_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[90]:


plt.imshow(uc10_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[91]:


body_wordcloud.to_file("body_wordcloud.png")
title_wordcloud.to_file("title_wordcloud.png")
uc1_wordcloud.to_file("uc1_wordcloud.png")
uc2_wordcloud.to_file("uc2_wordcloud.png")
uc3_wordcloud.to_file("uc3_wordcloud.png")
uc4_wordcloud.to_file("uc4_wordcloud.png")
uc5_wordcloud.to_file("uc5_wordcloud.png")
uc6_wordcloud.to_file("uc6_wordcloud.png")
uc7_wordcloud.to_file("uc7_wordcloud.png")
uc8_wordcloud.to_file("uc8_wordcloud.png")
uc9_wordcloud.to_file("uc9_wordcloud.png")
uc10_wordcloud.to_file("uc10_wordcloud.png")


# In[92]:


sns.set(style='whitegrid')
plt.figure(figsize=(20, 10))
body_sns_plot = sns.barplot(x="body_frequency", y = "body", color = 'lightblue', data = word_df[:50])


# In[93]:


plt.figure(figsize=(20, 10))
sns.set(style='whitegrid')
title_sns_plot = sns.barplot(x="title_frequency", y = "title",  color = 'lightblue', data = word_df[:50])


# In[94]:


sns.set(style="whitegrid")
plt.figure(figsize=(20, 10))
uc1_sns_plot = sns.barplot(x="uc1_frequency", y="uc1",  color = 'lightblue', data=word_df[:50])


# In[95]:


sns.set(style="whitegrid")
plt.figure(figsize=(20, 10))
uc2_sns_plot = sns.barplot(x="uc2_frequency", y="uc2", color = 'lightblue', data=word_df[:50])


# In[96]:


sns.set(style="whitegrid")
plt.figure(figsize=(20, 10))
uc3_sns_plot = sns.barplot(x="uc3_frequency", y="uc3", color = 'lightblue', data=word_df[:50])


# In[97]:


sns.set(style="whitegrid")
plt.figure(figsize=(20, 10))
uc4_sns_plot = sns.barplot(x="uc4_frequency", y="uc4",color = 'lightblue', data=word_df[:50])


# In[98]:


sns.set(style="whitegrid")
plt.figure(figsize=(20, 10))
uc5_sns_plot = sns.barplot(x="uc5_frequency", y="uc5", color = 'lightblue', data=word_df[:50])


# In[99]:


sns.set(style="whitegrid")
plt.figure(figsize=(20, 10))
uc6_sns_plot = sns.barplot(x="uc6_frequency", y="uc6", color = 'lightblue',data=word_df[:50])


# In[100]:


sns.set(style="whitegrid")
plt.figure(figsize=(20, 10))
uc7_sns_plot = sns.barplot(x="uc7_frequency", y="uc7", color = 'lightblue',data=word_df[:50])


# In[101]:


sns.set(style="whitegrid")
plt.figure(figsize=(20, 10))
uc8_sns_plot = sns.barplot(x="uc8_frequency", y="uc8",color = 'lightblue', data=word_df[:50])


# In[102]:


sns.set(style="whitegrid")
plt.figure(figsize=(20, 10))
uc9_sns_plot = sns.barplot(x="uc9_frequency", y="uc9",color = 'lightblue', data=word_df[:50])


# In[103]:


sns.set(style="whitegrid")
plt.figure(figsize=(20, 10))
uc10_sns_plot = sns.barplot(x="uc10_frequency", y="uc10", color = 'lightblue',data=word_df[:50])


# In[104]:


story_sns_plot.figure.savefig("story_wordbar.png")
title_sns_plot.figure.savefig("title_wordbar.png")
uc1_sns_plot.figure.savefig("uc1_wordbar.png")
uc2_sns_plot.figure.savefig("uc2_wordbar.png")
uc3_sns_plot.figure.savefig("uc3_wordbar.png")
uc4_sns_plot.figure.savefig("uc4_wordbar.png")
uc5_sns_plot.figure.savefig("uc5_wordbar.png")
uc6_sns_plot.figure.savefig("uc6_wordbar.png")
uc7_sns_plot.figure.savefig("uc7_wordbar.png")
uc8_sns_plot.figure.savefig("uc8_wordbar.png")
uc9_sns_plot.figure.savefig("uc9_wordbar.png")
uc10_sns_plot.figure.savefig("uc10_wordbar.png")


# In[ ]:





# In[ ]:




