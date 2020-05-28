#!/usr/bin/env python
# coding: utf-8

# In[11]:


import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import csv
import time


# In[12]:


counter = 1
url = 'https://old.reddit.com/r/AmItheAsshole/top/?sort=top&t=all'
headers = {'User-Agent': 'Mozilla/5.0'}
page = requests.get(url, headers=headers)
soup = BeautifulSoup(page.text, 'html.parser')
while counter <= 10:
    for post in soup.find_all('div', attrs={'class': 'thing', 'data-domain': 'self.AmItheAsshole'}):
        try:
            verdict = post.find("span", attrs={"class": "linkflairlabel"}).text
        except AttributeError:
            verdict = 'No-Flair'
        comment_button = soup.find("a", attrs={"data-event-action":"title"})
        for link in post.find_all('a', class_="comments", text=True):            
            comment_link = link['href']
            comment_page = requests.get(comment_link, headers=headers)
            comment_soup = BeautifulSoup(comment_page.text, 'html.parser')
            md_container = comment_soup.find_all("div", attrs={"class":"usertext-body may-blank-within md-container"})
            for content in md_container[1]:
                try:
                    story = content.text
                except AttributeError:
                    continue                    
            user_comments = {}
            stopper = 11
            if len(md_container) < stopper:
                stopper = len(md_container)
            for i in range(2, stopper):
                for content in md_container[i]:
                    try:
                        comment = content.text
                        user_comments["comment.{}".format(i)] = comment
                    except AttributeError:
                        continue                    
        for span in post:
            post.span.extract()
        title = post.find('p', class_="title").text
        try:
            author = post.find('a', class_='author').text
        except AttributeError:     
            author = 'DELETED'
        comments = post.find('a', class_='comments').text.split()[0]
        likes = post.find("div", attrs={"class": "score likes"}).text
        datetime = post.find("time")
        timestamp = datetime.get('datetime')
        if '-' in author[:1]:
            author.strip('-')
        post_line = [title, author, likes, comments, story, timestamp, verdict]
            
        for keys, values in user_comments.items():
            user_comment = values
            post_line.append(user_comment)
                    
        with open('reddit_aita.csv', 'a+', newline = '') as f:
            writer = csv.writer(f)
            writer.writerow(unicode(post_line).encode('utf-8'))
        print(counter)
        counter += 1
    next_button = soup.find("span", class_="next-button")
    next_page_link = next_button.find("a").attrs['href']
    time.sleep(2)
    page = requests.get(next_page_link, headers=headers)
    soup = BeautifulSoup(page.text, 'html.parser')


# In[21]:


names=['title', 'author', 'likes', 'comments', 'story', 'timestamp', 'verdict']
for i in range(1, 11):
    comment_number = "user_comment{}".format(i)
    names.append(comment_number)
reddit_aita = pd.read_csv('data/reddit_aita.csv', names=names)


# In[19]:


reddit_aita


# In[20]:


reddit_aita.to_csv('data/reddit_aitaR.csv')


# In[ ]:




