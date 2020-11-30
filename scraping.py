# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 17:17:46 2020

@author: Adam
"""


import re
import tweepy
import psycopg2
import pandas as pd
from textblob import TextBlob
import sys
import os


# Override tweepy.StreamListener to add logic to on_status
class MyStreamListener(tweepy.StreamListener):
    
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
        
        user_created_at = status.user.created_at
        user_description = status.user.description
        
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


conn = psycopg2.connect(DATABASE_URL, sslmode='require')
cur = conn.cursor()


special_words = 'Biden'
            
# HERE TOKENS ARE HIDEEN
API_KEY = ""
API_SECRET_KEY = ""
ACCESS_TOKEN = ""
ACCESS_TOKEN_SECRET = ""

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
    
conn.close()


