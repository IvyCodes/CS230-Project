# reddit-aita
Analysis of r/AmItheAsshole

# Overview
Reddit is essentially a massive forum discussion board where users can find a subreddit of their niche or interests. Users can submit or view posts from other users and can interact via comments, upvoting or downvoting. Upvoting a post or comments gives the comment a score, the more upvotes the post/comment has the higher the visibility. Downvoting does the reverse effect and usually down when the user disagrees with the post. The subreddit AmItheAsshole(AITA) allows users to post a story where the user is uncertain if they are the bad guys in the scenario. Then users can comment on these post with either You're-The-Asshole (YTA), Not-The-Asshole(NTA), not enough info(INFO), Everyone-Sucks-Here(ESH), No-Asshole-Here(NAH) and some reasoning behind their decision. Then the highest upvoted comment will be the subreddit decision and then the post will be flagged with one of the 5 Judgements. 

This creates an interesting dynamic where we can gain some insights on how average human beings think during what they preceive as a moral dilemma. The objectives of this project is to gather insights/themes of what the subreddit of 1.2million subscribers think and determine if there is common themes that these users upvote. To achieve this, I have scrapped the top 1000 post of all time (by upvotes) off reddit with the following information.

| Title | Author | Timestamp | Story | Upvotes | Flair/outcome | Top User comments 1-10 | 
| --- | --- | --- | --- | --- | --- | --- |

# Business Understanding
With these information and insights, it can also provide businesses on how individual feels about moral dilemma. It is possible that we can create a model to predict if the employee was morally incorrect in the situation. This could assist Human Resources(HR) or Management to have assistance on whether or not if the employee is in the right or wrong. It shouldn't be used as the deciding factor but rather something to help them lead to a decision.

We will also be creating a text based generator model that will hopefully create a more fluid story telling bot. This could be useful for a business as it can help the HR create moral scenarios to help screen candidates on how they precieve moral dilemmas.

# Expectation

List of expectation of what the insights will be:
- Determine what the subreddit common outcomes are. E.g. Is the subreddit more inclined to say YTA or NTA.
- Determine the common words. 
- Seperate and determine common themes. E.g. Are more top post about family, relationship, work or just general interactions etc.
- Determine if the top 10 comments by majority conveys a different outcome. As the outcome is done by the decision of the highest upvoted comment, it is possible that users initally upvoted that post highly and it got more visibility early. Therefore a possibilty that the outcome could be different.
- Determine if there is common theme to the commenters judgement.
- Determine if it is possible to find seasonilty to votes. As certain seasons, users could be more inclined to YTA or NTA.
- Determine if we can identify if the subreddit is political or not. E.g. Is it possible for it to be left or right wing politically.
- Have enough data to try a text generator model to create a post and see if it can be popular.
- Create a model that can predict the judgement of the posts. 

# Risks/Difficulties

There could be a bunch of potential risks and difficulties for this project.
- Incorrect sampling: As we are getting the top 1000 posts, it is possible that there is some bias to the sample even though the sample is statisically significant. This is because we are getting the top 1000 posts and also not gathering the top 1000 most downvoted posts.
- Modelling issues: Lack of data/low amount of data could make the model unable to generate a text. An uneven distribution of judgements such as low amount of posts with INFO. This could make it difficult for the model to predict more information required. 

# Project life-cycle and current stages
Essentially following CRISP-DM methodology. We won't be doing a deployment phase as there really isn't anything to deploy.

- [x] **Project and Business understanding**: Understand what the objectives, expectations and possible pitfalls of the project will be.
- [x] **Data Acquisition**: Scrap the data from the subreddit
- [ ] **Data cleaning/Preparation** - *Current Stage*: Clean the data as most of the text will required to be standardised and remove filler words.
- [ ] **Insights creation/dashboards**: Create general visualisation of the data at hand. E.g. a wordmap, bar graphs etc.
- [ ] **Data Modelling**: Model a text generator to create a AITA story
- [ ] **Evaluation**: Determine if the model was successful and how the overall project went

# What's in the Repo?

In the repo there is:
- reddit_scrapper: The Jupyter Notebook for the reddit scraping bot using BeautifulSoup. I could of just used PRAW(Python Reddit API Wrapper) but for educational purposes I have chosen a webscrapper to learn how it works. 
- reddit_aita: No header/column name comma separated values(csv) file of the webscrapping.
- reddit_aitaR: With header/column name comma separated values(csv) file of the webscrapping.

# What's the specification of my machine?

Computational time will be different if you clone and run this on a different machine. 

My computer spec: 

| CPU | RAM | GPU |
| -- | -- | -- |
| AMD Ryzen 7 3700x | (2x8gb) 16gb DDRR4 3200MHZ dual channel | AMD RX 480 |
