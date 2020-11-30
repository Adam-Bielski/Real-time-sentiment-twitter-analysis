# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 17:17:46 2020

@author: Adam
"""

# This is Main function.
# Extracting streaming data from Twitter, pre-processing, and loading into MySQL
# import credentials # Import api/access_token keys from credentials.py
# import settings # Import related setting constants from settings.py 

import re
import tweepy
# import mysql.connector
import psycopg2
import pandas as pd
from textblob import TextBlob
import sys
import os
# sys.path.insert(1, R'C:\Users\Adam\Desktop\Twitter_application')
#import TWITTER_KEYS

# Streaming With Tweepy 
# http://docs.tweepy.org/en/v3.4.0/streaming_how_to.html#streaming-with-tweepy


# Override tweepy.StreamListener to add logic to on_status
class MyStreamListener(tweepy.StreamListener):
    '''
    Tweets are known as “status updates”. So the Status class in tweepy has properties describing the tweet.
    https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/tweet-object.html
    '''
    
    def on_status(self, status):
        '''
        Extract info from tweets
        '''
        if status.retweeted:
            # Avoid retweeted info, and only original tweets will be received
            return True
        # Extract attributes from each tweet
        id_str = status.id_str
        created_at = status.created_at
        text = status.text  # Pre-processing the text  
        # sentiment = TextBlob(text).sentiment
        # polarity = sentiment.polarity
        # subjectivity = sentiment.subjectivity
        
        user_created_at = status.user.created_at
        # user_location = status.user.location
        user_description = status.user.description
        # user_followers_count =status.user.followers_count
        # longitude = cleaning(status.user.location)
        # latitude = cleaning(status.user.location)
        # if status.coordinates:
        #     longitude = status.coordinates['coordinates'][0]
        #     latitude = status.coordinates['coordinates'][1]
            
        # retweet_count = status.retweet_count
        # favorite_count = status.favorite_count
        
        cur = conn.cursor()
        sql = "INSERT INTO TWITTER (id_str, created_at, text, user_created_at, user_description) VALUES (%s, %s, %s, %s, %s)"
        val = (id_str, created_at, text, user_created_at, user_description)
        cur.execute(sql, val)
        conn.commit()
        
        delete_query = '''
        DELETE FROM TWITTER
        WHERE id_str IN (
            SELECT id_str 
            FROM TWITTER
            ORDER BY created_at asc
            LIMIT 200) AND (SELECT COUNT(*) FROM TWITTER) > 9600;
        '''
        
        cur.execute(delete_query)
        conn.commit()
        cur.close()
    
    
    def on_error(self, status_code):
        '''
        Since Twitter API has rate limits, stop srcraping data as it exceed to the thresold.
        '''
        if status_code == 420:
            # return False to disconnect the stream
            return False
        
    def on_exception(self, exception):
       print(exception)
       return
        
        
def cleaning(text):                  
    if text:
        return text.encode('ascii', 'ignore').decode('ascii')                           
    
    
        
        
        





table_val_types = "id_str VARCHAR(1000), created_at VARCHAR(1000), text VARCHAR(1000), user_created_at VARCHAR(1000), user_description VARCHAR(1000)"

DATABASE_URL = os.environ['DATABASE_URL']
# DATABASE_URL = "postgres://zhpcypxfkrbgpo:b4b9d6f7358bf27c5c1a85172478f4b9b635641fc608389733a18de09f9aa1d7@ec2-176-34-114-78.eu-west-1.compute.amazonaws.com:5432/d9961albf0ani7"


conn = psycopg2.connect(DATABASE_URL, sslmode='require')
cur = conn.cursor()


special_words = 'Biden'
            
# auth  = tweepy.OAuthHandler(TWITTER_KEYS.API_KEY, TWITTER_KEYS.API_SECRET_KEY)
# auth.set_access_token(credentials.ACCESS_TOEKN, credentials.ACCESS_TOKEN_SECRET)

API_KEY = "v9HXhDuEIZdLLqwbWHMhCZqBH"
API_SECRET_KEY = "Y1CNSC9CUxU9m8y4Lzr0kyM21NdkDj9q0gsK5uQG7dNGPpdgJ7"
ACCESS_TOKEN = "1330444403199119362-UemFzYigrHNrAfLFawQtlhA5YJzooJ"
ACCESS_TOKEN_SECRET = "4DywopQDwHfWNpihnGmx9hDkO5ucb7TTSN4IGaqL5s5Df"

auth  = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)


myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth = api.auth, listener = myStreamListener)

while True:
    try:
        myStream.filter(languages=["en"],track=special_words)
    except:
        continue
    
    

# Close the MySQL connection as it finished
# However, this won't be reached as the stream listener won't stop automatically
# Press STOP button to finish the process.
conn.close()


