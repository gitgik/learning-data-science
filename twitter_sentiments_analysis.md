
#### Twitter Sentiments Analyzer
Twitter is a treasure trove of sentiments. People output thousands of reaction and opinions on pretty much any topic.

We'll create a twitter sentiments analyzer which will have the ability to determin how positive or negative a tweets emotion is.

The process:
* We break down the tweets into small tokens and count the number of times each word shows up.
* Then we look for the sentiments value of each word from the sentiments lexicon and classify the total sentiment value of our tweet.

The steps are as follows:
* Register for Twitter API
* Install dependencies (pip install tweepy)
* Write the script



```python
import os
import tweepy
from textblob import TextBlob

consumer_key = os.getenv("CONSUMER_KEY")
consumer_secret = os.getenv("CONSUMER_SECRET")

access_token = os.getenv("ACCESS_TOKEN")
access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")

# authenticate with the Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# set access token
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# search for some tweets
public_tweets = api.search("Trump")

# loop through tweets and display their sentiments
for tweet in public_tweets:
    print (tweet.text)
    
    analysis = TextBlob(tweet.text)
    print (analysis.sentiment)

```
