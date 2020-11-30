# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:57:15 2020

@author: Adam
"""


  
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import itertools
import math
import base64
from flask import Flask
import os
import psycopg2
import datetime
 
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from textblob import TextBlob
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Twitter-dashbourd'

server = app.server

app.layout = html.Div(children=[
    html.H2('Real-time twitter dashbourd of tweets containing word "Biden"', style={
        'textAlign': 'left'
    }),
    html.H4('Created by: Adam Bielski \n November 2020', style={
        'textAlign': 'right'
    }),
    
    html.Div(id='live-update-graph'),
    html.Div(id='live-update-graph-bottom'),
        
           
    
    # html.Img(id="image_wc"),
    html.Div(id="recent-tweets-table"),
    
    html.Br(),
    
    
    

    dcc.Interval(
        id='interval-component',
        interval=1*100000, # in milliseconds
        n_intervals=0
    )
    ], style={'padding': '20px'})







# Multiple components can update everytime interval gets fired.
@app.callback(Output('live-update-graph', 'children'),
              [Input('interval-component', 'n_intervals')])
def update_graph_live(n):

   
    
    
    # Getting only twitters from the last 10 minutes
    time_now = datetime.datetime.utcnow()
    time_10mins_before = datetime.timedelta(hours=3,minutes=0)
    time_interval = time_now - time_10mins_before
    
    
    # Loading data from Heroku PostgreSQL
    DATABASE_URL = os.environ['DATABASE_URL']
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    query = "SELECT id_str, created_at, text, user_created_at, user_description FROM TWITTER WHERE created_at >= '{}' AND lower(text) LIKE '%biden%'".format(time_interval)
    df = pd.read_sql(query, con=conn)
    

    # Convert UTC into PDT
    df['created_at'] = pd.to_datetime(df['created_at'])
        
    
    def cleaning(text):
        # Getting rid of emojis                  
        if text:
            text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w+:\/\/\S+)", " ", text).split())
            text = text.encode('ascii', 'ignore').decode('ascii')
            text = re.sub(r"http\S+", "", text)
            text = text.replace('RT ', ' ').replace('&amp;', 'and')
            text = re.sub('[!"#$%&\\(\\)\\*\\+,-\\./:;<=>\\?@\\[\\]\\\\^_`\\{\\|\\}~]', '', text)
            text = re.sub('[0-9]', '', text)
            text = re.sub("\S+\\'\S+", '', text)
            text = re.sub("\s+", ' ', text)
            text = text.lower()
            text = str(text)
            return text                    
    
    
    def polarity(text):
        if text:
            sentiment = TextBlob(text).sentiment
            polarity = sentiment.polarity
            return polarity
        
        
    def subjectivity(text):
        if text:
            sentiment = TextBlob(text).sentiment
            subjectivity = sentiment.subjectivity
            return subjectivity
        
    
    # Gettin rid of unnecessary non-word characters and emojis
    # Calculating polarity and subjectivity of each tweet
    df['text'] = df['text'].apply(cleaning)
    df['text'].replace('', np.nan, inplace=True)
    df.dropna(inplace=True, subset=['text'], axis=0)
    df['user_description'] = df['user_description'].apply(cleaning)
    df['user_description'].replace('', np.nan, inplace=True)
    df.dropna(inplace=True, subset=['user_description'], axis=0)
    df['polarity'] = pd.cut(df['text'].apply(polarity), 3)
    df['subjectivity'] = pd.cut(df['text'].apply(subjectivity), 2)  
    
    # Grouping tweets to every 15 sec
    result = df.groupby([pd.Grouper(key='created_at', freq='300s'), 'polarity', 'subjectivity']).count().unstack(fill_value=0).stack().reset_index()
    result = result.rename(columns={ "id_str": "Num of 'Facebook' mentions","created_at":"Time in UTC" })
    time_series = result["Time in UTC"][result['polarity']==result['polarity'][0]].reset_index(drop=True)


    neu = result["Num of 'Facebook' mentions"][ result['polarity']==np.unique(df['polarity'])[1]].sum()
    pos = result["Num of 'Facebook' mentions"][ result['polarity']==np.unique(df['polarity'])[2]].sum()
    neg = result["Num of 'Facebook' mentions"][result['polarity']==np.unique(df['polarity'])[0]].sum()
    

    percent=0.2
    daily_impressions = 5000
    daily_tweets_num =5000
    
    
    children = [
                html.Div([
                    html.Div([
                        dcc.Graph(
                            id='crossfilter-indicator-scatter',
                            figure={
                                'data': [
                                    go.Bar(
                                        x=time_series,
                                        y=result["Num of 'Facebook' mentions"][result['polarity']==np.unique(df['polarity'])[1]].reset_index(drop=True),
                                        name="Neutrals",
                                        opacity=0.8,
                                        marker_color='rgb(212, 212, 212)'
                                    ),
                                    go.Bar(
                                        x=time_series,
                                        y=result["Num of 'Facebook' mentions"][result['polarity']==np.unique(df['polarity'])[0]].reset_index(drop=True),
                                        name="Negatives",
                                        opacity=0.8,
                                        marker_color='rgb(237, 0, 38)'
                                    ),
                                    go.Bar(
                                        x=time_series,
                                        y=result["Num of 'Facebook' mentions"][result['polarity']==np.unique(df['polarity'])[2]].reset_index(drop=True),
                                        name="Positives",
                                        opacity=0.8,
                                        marker_color='rgb(207, 151, 215)'
                                    )
                                ],
                                'layout':{'title':'Sentiments of tweets across time'}
                            }
                        )
                    ], style={'width': '73%', 'display': 'inline-block', 'padding': '0 0 0 20'}),
                    
                    html.Div([
                        dcc.Graph(
                            id='pie-chart',
                            figure={
                                'data': [
                                    go.Treemap(
                                        labels=['Positive', 'Negative','Neutral'], 
                                        values=[pos, neg, neu],
                                        parents=['All', 'All', 'All'],
                                        name="View Metrics",
                                        marker_colors = ['rgb(207, 151, 215)', 'rgb(237, 0, 38)', 'rgb(212, 212, 212)'],
                                        textinfo = "label+percent parent",branchvalues="total")
                                ],
                                'layout':{
                                    'title':'Sentiments of tweets in the last 15 minutes'
                                }

                            }
                        )
                    ], style={'width': '27%', 'display': 'inline-block'})
                ]),
        ]    
        
    return children



@app.callback(Output('recent-tweets-table', 'children'),
              [Input('interval-component', 'n_intervals')])        
def update_recent_tweets(sentiment_term):

    # db_connection = mysql.connector.connect(
    # host="localhost",
    # user="root",
    # passwd="Haselko22",
    # database="Twitter",
    # charset = 'utf8')
    
    
    # Getting only twitters from the last 10 minutes
    time_now = datetime.datetime.utcnow()
    time_10mins_before = datetime.timedelta(hours=0,minutes=30)
    time_interval = time_now - time_10mins_before
    
    # Loading data from Heroku PostgreSQL
    DATABASE_URL = os.environ['DATABASE_URL']
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    query = "SELECT id_str, created_at, text, user_created_at, user_description FROM TWITTER WHERE created_at >= '{}' AND lower(text) LIKE '%biden%'".format(time_interval)
    df = pd.read_sql(query, con=conn)
    
    def generate_table(df, max_rows=10):
        return html.Table(className="responsive-table",
                          children=[
                              html.Thead(html.Tr(children=[html.Th(col.title()) for col in df.columns.values])),
                              html.Tbody(
                                  [html.Tr(children=[html.Td(data) for data in d]) for d in df.values.tolist()])])
    


    df_reduced = df[['created_at','text']]
    df.rename(columns={'created_at':'Time', 'text':'Text of tweet'})

    return generate_table(df_reduced, max_rows=10)







if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run_server(debug=False, host='0.0.0.0', port=port)
