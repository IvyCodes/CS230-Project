#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import praw
from psaw import PushshiftAPI
import datetime as dt


# In[2]:


reddit = praw.Reddit(client_id='ha7wPvCUY_DurA',
                     client_secret='B98E34rnXS-Qw6Fg4eOwHUpIupQ',
                     user_agent='aita_scrapper'
                    )

api = PushshiftAPI(reddit)



end_epoch=int(dt.datetime(2019, 1, 2).timestamp())
start_epoch=int(dt.datetime(2019, 1, 1).timestamp())
counter = 0

for i in range(365):
    for post in list(api.search_submissions(before=end_epoch, after = start_epoch, subreddit = 'AmItheAsshole', limit = 100, filter = ['id', 'title', 'score',  'num_comments', 'selftext', 'created', 'link_flair_text'])):
        post = [[post.id, post.title, post.score,  post.num_comments, post.selftext, post.created, post.link_flair_text]]
        aita_df = pd.DataFrame(post, columns = ['id', 'title', 'score', 'num_comments', 'body', 'created', 'flair'])
        # print(aita_df)
        aita_df.to_csv('aita.csv', mode = 'a+', index = None, header = None)
        counter += 1
    print('processed ' + str(counter) + ' entries')
    start_epoch +=  86400
    end_epoch +=  86400


