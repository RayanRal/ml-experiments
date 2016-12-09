import tweepy
from textblob import TextBlob

# Step 1 - Authenticate
consumer_key = 'XWjj85X2NNCcvEBmTL9Oyooj8'
consumer_secret = 'PPAWuc3Z9EhfnO3LE5i2Dql1ttj7wTv1yJwLdtaoD9CyPs33pp'

access_token = '98723802-mBCapmb6Pl502Ch93UlxOgQY6vPZl5zUSSg4PaQ7G'
access_token_secret = '4ZiDgc0BRXJFw6WWWjhalaw3EB7LelKhkYmKQdUNendd6'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# Step 3 - Retrieve Tweets
public_tweets = api.search('Trump')

# CHALLENGE - Instead of printing out each tweet, save each Tweet to a CSV file
# and label each one as either 'positive' or 'negative', depending on the sentiment
# You can decide the sentiment polarity threshold yourself


for tweet in public_tweets:
    print(tweet.text)

    # Step 4 Perform Sentiment Analysis on Tweets
    analysis = TextBlob(tweet.text)

    print(analysis.sentiment)
    print("")