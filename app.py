# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:57:15 2020

@author: Adam
"""
# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Input, Output
# import pandas as pd
# import plotly.graph_objs as go
# # import settings
# import itertools
# import math
# import base64
# from flask import Flask
# import os
# import psycopg2
# import datetime
# # import mysql.connector
# import numpy as np
# # from sklearn.feature_extraction.text import CountVectorizer
# # from sklearn.decomposition import LatentDirichletAllocation
# import plotly.express as px
# import dash.dependencies as dd
# import dash_core_components as dcc
# import dash_html_components as html
# from io import BytesIO
# from wordcloud import WordCloud
# import base64
# import urllib


  
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
        
        
    # html.Div(className='row',
    #           children=[html.Div(id="recent-tweets-table"), html.Img(id="image_wc")], style={'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '3vw', 'margin-top': '3vw'}),
    # html.Div(className='row', children=[html.Div(id="recent-tweets-table", className='col s12 m6 l6'),
    #                                 html.Div(html.Img(id='image_wc'), className='col s12 m6 l6'),]),   
    
    
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

    
    # db_connection = mysql.connector.connect(
    # host="localhost",
    # user="root",
    # passwd="Haselko22",
    # database="Twitter",
    # charset = 'utf8')
    
    
    # Getting only twitters from the last 10 minutes
    time_now = datetime.datetime.utcnow()
    time_10mins_before = datetime.timedelta(hours=3,minutes=0)
    time_interval = time_now - time_10mins_before
    
    
    # Loading data from Heroku PostgreSQL
    DATABASE_URL = os.environ['DATABASE_URL']
    # DATABASE_URL = "postgres://zhpcypxfkrbgpo:b4b9d6f7358bf27c5c1a85172478f4b9b635641fc608389733a18de09f9aa1d7@ec2-176-34-114-78.eu-west-1.compute.amazonaws.com:5432/d9961albf0ani7"
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    query = "SELECT id_str, created_at, text, user_created_at, user_description FROM TWITTER WHERE created_at >= '{}' AND lower(text) LIKE '%biden%'".format(time_interval)
    # query = "SELECT id_str, created_at, text, user_created_at, user_description FROM TWITTER WHERE created_at >= '{}'".format(time_interval)
    df = pd.read_sql(query, con=conn)
    
    # Extracting tweets into dataframe
    # query = "SELECT id_str, text, created_at, polarity, user_location, user_description FROM TWITTER WHERE created_at >= '{}' AND lower(text) LIKE '%amazon%'".format(time_interval)
    # # query = "SELECT id_str, text, created_at, polarity, user_location, user_description FROM TWITTER"
    # df = pd.read_sql(query, con=db_connection)


    # Convert UTC into PDT
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    # df = df.iloc[:500, :]
    
    
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
    # df['subjectivity'] = df['text'].apply(subjectivity)
    df['polarity'] = pd.cut(df['text'].apply(polarity), 3)
    df['subjectivity'] = pd.cut(df['text'].apply(subjectivity), 2)  
    
    # Grouping tweets to every 15 sec
    result = df.groupby([pd.Grouper(key='created_at', freq='300s'), 'polarity', 'subjectivity']).count().unstack(fill_value=0).stack().reset_index()
    result = result.rename(columns={ "id_str": "Num of 'Facebook' mentions","created_at":"Time in UTC" })
    time_series = result["Time in UTC"][result['polarity']==result['polarity'][0]].reset_index(drop=True)


    # min10 = datetime.datetime.now() - datetime.timedelta(hours=7, minutes=5)
    # min20 = datetime.datetime.now() - datetime.timedelta(hours=7, minutes=15)

    neu = result["Num of 'Facebook' mentions"][ result['polarity']==np.unique(df['polarity'])[1]].sum()
    pos = result["Num of 'Facebook' mentions"][ result['polarity']==np.unique(df['polarity'])[2]].sum()
    neg = result["Num of 'Facebook' mentions"][result['polarity']==np.unique(df['polarity'])[0]].sum()
    
    # Loading back-up summary data
    # query = "SELECT daily_user_num, daily_tweets_num, impressions FROM Back_Up;"
    # # back_up = pd.read_sql(query, con=conn)
    # back_up = pd.read_sql(query, con=db_connection)
    # daily_tweets_num = back_up['daily_tweets_num'].iloc[0] + result[-6:-3]["Num of 'Facebook' mentions"].sum()
    # daily_impressions = back_up['impressions'].iloc[0] + df[df['created_at'] > (datetime.datetime.now() - datetime.timedelta(hours=7, seconds=10))]['user_followers_count'].sum()
    # cur = conn.cursor()

    # PDT_now = datetime.datetime.now() - datetime.timedelta(hours=7)
    # if PDT_now.strftime("%H%M")=='0000':
    #     cur.execute("UPDATE Back_Up SET daily_tweets_num = 0, impressions = 0;")
    # else:
    #     cur.execute("UPDATE Back_Up SET daily_tweets_num = {}, impressions = {};".format(daily_tweets_num, daily_impressions))
    # conn.commit()
    # cur.close()
    # conn.close()

    # Percentage Number of Tweets changed in Last 10 mins

    # count_now = df[df['created_at'] > min10]['id_str'].count()
    # count_before = df[ (min20 < df['created_at']) & (df['created_at'] < min10)]['id_str'].count()
    # percent = (count_now-count_before)/count_before*100
    # Create the graph 
   
    
    # vectorizer = CountVectorizer(max_df=0.9, min_df=10, token_pattern='\w+|\$[\d\.]+|\S+', stop_words='english', lowercase = False)

    # # apply transformation
    # tf = vectorizer.fit_transform(df['user_description']).toarray()
    
    # # tf_feature_names tells us what word each column in the matric represents
    # tf_feature_names = vectorizer.get_feature_names()
    
    
    
    
    
    # model = LatentDirichletAllocation(n_components=5, random_state=0)
    
    # model.fit(tf)
    
    # def display_topics(model, feature_names, no_top_words):
    #     topic_dict = {}
    #     for topic_idx, topic in enumerate(model.components_):
    #         topic_dict["Topic words-%d" % (topic_idx)]= ['{}'.format(feature_names[i])
    #                         for i in topic.argsort()[:-no_top_words - 1:-1]]
    #     return pd.DataFrame(topic_dict)
    
    
    # def display_weights(model, feature_names, no_top_words):
    #     topic_dict = {}
    #     for topic_idx, topic in enumerate(model.components_):
    #         topic_dict["Topic weights-%d" % (topic_idx)]= ['{:.1f}'.format(topic[i])
    #                         for i in topic.argsort()[:-no_top_words - 1:-1]]
    #     return pd.DataFrame(topic_dict)
    
    
    # no_top_words = 5
    # topics = display_topics(model, tf_feature_names, no_top_words)
    # topics.reset_index(inplace=True)
    # weights = display_weights(model, tf_feature_names, no_top_words)
    # weights.reset_index(inplace=True)
    
    # topics = pd.wide_to_long(topics, stubnames=['Topic words'], i='index',j='topic_number', sep='-')
    # weights = pd.wide_to_long(weights, stubnames=['Topic weights'], i='index',j='topic_number', sep='-')
    # topics = topics.merge(weights, left_index=True, right_index=True)
    # topics.reset_index(inplace=True)
    
    
    # treemap = px.treemap(topics, path=[px.Constant('Topics'), 'topic_number', 'Topic words'], values='Topic weights')
    # plot(treemap)
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
    # DATABASE_URL = "postgres://zhpcypxfkrbgpo:b4b9d6f7358bf27c5c1a85172478f4b9b635641fc608389733a18de09f9aa1d7@ec2-176-34-114-78.eu-west-1.compute.amazonaws.com:5432/d9961albf0ani7"
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    query = "SELECT id_str, created_at, text, user_created_at, user_description FROM TWITTER WHERE created_at >= '{}' AND lower(text) LIKE '%biden%'".format(time_interval)
    # query = "SELECT id_str, created_at, text, user_created_at, user_description FROM TWITTER WHERE created_at >= '{}'".format(time_interval)
    df = pd.read_sql(query, con=conn)
    
    def generate_table(df, max_rows=10):
        return html.Table(className="responsive-table",
                          children=[
                              html.Thead(html.Tr(children=[html.Th(col.title()) for col in df.columns.values])),
                              html.Tbody(
                                  [html.Tr(children=[html.Td(data) for data in d]) for d in df.values.tolist()])])
    
    # Extracting tweets into dataframe
    # query = "SELECT id_str, text, created_at, polarity, user_location, user_description FROM TWITTER WHERE created_at >= '{}' AND lower(text) LIKE '%amazon%'".format(time_interval)
    # # query = "SELECT id_str, text, created_at, polarity, user_location, user_description FROM TWITTER"
    # df = pd.read_sql(query, con=db_connection)



    df_reduced = df[['created_at','text']]
    df.rename(columns={'created_at':'Time', 'text':'Text of tweet'})

    return generate_table(df_reduced, max_rows=10)



# @app.callback(Output('live-update-graph-bottom', 'children'),
#               [Input('interval-component', 'n_intervals')])
# def update_graph_bottom_live(n):

#     # Loading data from Heroku PostgreSQL
#     # DATABASE_URL = os.environ['DATABASE_URL']
#     # conn = psycopg2.connect(DATABASE_URL, sslmode='require')
#     # query = "SELECT id_str, text, created_at, polarity, user_location FROM TWITTER
#     # "
#     # df = pd.read_sql(query, con=conn)
#     # conn.close()
    
#     db_connection = mysql.connector.connect(
#     host="localhost",
#     user="root",
#     passwd="Haselko22",
#     database="Twitter",
#     charset = 'utf8')
    
    
#     # Getting only twitters from the last 10 minutes
#     time_now = datetime.datetime.utcnow()
#     time_10mins_before = datetime.timedelta(hours=0,minutes=2)
#     time_interval = time_now - time_10mins_before
    
#     # Extracting tweets into dataframe
#     query = "SELECT id_str, text, created_at, polarity, user_location, user_description FROM TWITTER WHERE created_at >= '{}'".format(time_interval)
#     # query = "SELECT id_str, text, created_at, polarity, user_location, user_description FROM TWITTER"
#     df = pd.read_sql(query, con=db_connection)



#     # Convert UTC into PDT
#     df['created_at'] = pd.to_datetime(df['created_at']).apply(lambda x: x - datetime.timedelta(hours=7))

#     # Clean and transform data to enable word frequency
#     def cleaning(text):
#         # Getting rid of emojis                  
#         if text:
#             text = str(text)
#             text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w+:\/\/\S+)", " ", text).split())
#             text = text.encode('ascii', 'ignore').decode('ascii')
#             text = re.sub(r"http\S+", "", text)
#             text = text.replace('RT ', ' ').replace('&amp;', 'and')
#             text = re.sub('[!"#$%&\\(\\)\\*\\+,-\\./:;<=>\\?@\\[\\]\\\\^_`\\{\\|\\}~]', '', text)
#             text = re.sub('[0-9]', '', text)
#             text = re.sub("\S+\\'\S+", '', text)
#             text = re.sub("\s+", ' ', text)
#             text = text.lower()
#             return text
        
        
#     df['text'] = df['text'].apply(cleaning)
#     df['text'].replace('', np.nan, inplace=True)
#     content = df['text'].dropna()
#     content = content.apply(str)
#     content = np.array(content.array)
#     content = list(filter(lambda x: len(x)>5, content))
#     content = list(map(lambda x: x.replace('"', "'"), content))
#     content = ' '.join(content)


#     tokenized_word = word_tokenize(content)
#     stop_words=set(stopwords.words("english"))
#     filtered_sent=[]
#     for w in tokenized_word:
#         if (w not in stop_words) and (len(w) >= 3):
#             filtered_sent.append(w)
#     fdist = FreqDist(filtered_sent)
#     fd = pd.DataFrame(fdist.most_common(8), columns = ["Word","Frequency"]).reindex()
#     fd['Polarity'] = fd['Word'].apply(lambda x: TextBlob(x).sentiment.polarity)
#     fd['Marker_Color'] = fd['Polarity'].apply(lambda x: 'rgba(255, 50, 50, 0.6)' if x < -0.1 else \
#         ('rgba(184, 247, 212, 0.6)' if x > 0.1 else 'rgba(131, 90, 241, 0.6)'))
#     fd['Line_Color'] = fd['Polarity'].apply(lambda x: 'rgba(255, 50, 50, 1)' if x < -0.1 else \
#         ('rgba(184, 247, 212, 1)' if x > 0.1 else 'rgba(131, 90, 241, 1)'))
#     dfm = fd[['Word', 'Frequency']]
#     dfm.rename(columns={'Word':'word', 'Frequency':'frequency'}, inplace=True)
        


    



#     # Create the graph 
#     children = [
#                 html.Div([
#                     dcc.Graph(
#                         id='x-time-series',
#                         figure = {
#                             'data':[
#                                 go.Bar(                          
#                                     x=fd["Frequency"].loc[::-1],
#                                     y=fd["Word"].loc[::-1], 
#                                     name="Neutrals", 
#                                     orientation='h',
#                                     marker_color=['rgb(31, 66, 96)']*8,
#                                 )
#                             ],
#                             'layout':{
#                                 'hovermode':"closest", 'title':'The most common words'}
#                             }
#                     )
#                 ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 0 0 20'})
#             ]
        
#     return children


    
    
    
# @app.callback(Output('image_wc', 'src'),
#           [Input('interval-component', 'n_intervals')])
# def update_worldcloud(n):
#     # db_connection = mysql.connector.connect(
#     # host="localhost",
#     # user="root",
#     # passwd="Haselko22",
#     # database="Twitter",
#     # charset = 'utf8')
    
    
    
#     # # Getting only twitters from the last 10 minutes
#     # time_now = datetime.datetime.utcnow()
#     # time_10mins_before = datetime.timedelta(hours=0,minutes=30)
#     # time_interval = time_now - time_10mins_before
    
    
#     # Loading data from Heroku PostgreSQL
#     DATABASE_URL = os.environ['DATABASE_URL']
#     # DATABASE_URL = "postgres://zhpcypxfkrbgpo:b4b9d6f7358bf27c5c1a85172478f4b9b635641fc608389733a18de09f9aa1d7@ec2-176-34-114-78.eu-west-1.compute.amazonaws.com:5432/d9961albf0ani7"
#     conn = psycopg2.connect(DATABASE_URL, sslmode='require')
#     query = "SELECT id_str, created_at, text, user_created_at, user_description FROM TWITTER WHERE created_at >= '{}' AND lower(text) LIKE '%biden%'".format(time_interval)
#     # query = "SELECT id_str, created_at, text, user_created_at, user_description FROM TWITTER WHERE created_at >= '{}'".format(time_interval)
#     df = pd.read_sql(query, con=conn)
    
#     # # Extracting tweets into dataframe
#     # query = "SELECT id_str, text, created_at, polarity, user_location, user_description FROM TWITTER WHERE created_at >= '{}' AND lower(text) LIKE '%amazon%'".format(time_interval)
#     # # query = "SELECT id_str, text, created_at, polarity, user_location, user_description FROM TWITTER"
#     # df = pd.read_sql(query, con=db_connection)
    
#     def cleaning(text):
#         # Getting rid of emojis                  
#         if text:
#             text = str(text)
#             text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w+:\/\/\S+)", " ", text).split())
#             text = text.encode('ascii', 'ignore').decode('ascii')
#             text = re.sub(r"http\S+", "", text)
#             text = text.replace('RT ', ' ').replace('&amp;', 'and')
#             text = re.sub('[!"#$%&\\(\\)\\*\\+,-\\./:;<=>\\?@\\[\\]\\\\^_`\\{\\|\\}~]', '', text)
#             text = re.sub('[0-9]', '', text)
#             text = re.sub("\S+\\'\S+", '', text)
#             text = re.sub("\s+", ' ', text)
#             text = text.lower()
#             return text
        
#     def plot_wordcloud(data):
#         d = {a: x for a, x in data.values}
#         wc = WordCloud(background_color='black', width=480, height=360)
#         wc.fit_words(d)
#         return wc.to_image()
        
    
#     df['text'] = df['text'].apply(cleaning)
#     df['text'].replace('', np.nan, inplace=True)
#     content = df['text'].dropna()
#     content = content.apply(str)
#     content = np.array(content.array)
#     content = list(filter(lambda x: len(x)>5, content))
#     content = list(map(lambda x: x.replace('"', "'"), content))
#     content = ' '.join(content)


#     tokenized_word = content.split(' ')
#     stop_words=set(stopwords.words("english"))
#     filtered_sent=[]
#     for w in tokenized_word:
#         if (w not in stop_words) and (len(w) >= 3):
#             filtered_sent.append(w)
#     fdist = FreqDist(filtered_sent)
#     fd = pd.DataFrame(fdist.most_common(8), columns = ["Word","Frequency"]).reindex()
#     fd['Polarity'] = fd['Word'].apply(lambda x: TextBlob(x).sentiment.polarity)
#     fd['Marker_Color'] = fd['Polarity'].apply(lambda x: 'rgba(255, 50, 50, 0.6)' if x < -0.1 else \
#         ('rgba(184, 247, 212, 0.6)' if x > 0.1 else 'rgba(131, 90, 241, 0.6)'))
#     fd['Line_Color'] = fd['Polarity'].apply(lambda x: 'rgba(255, 50, 50, 1)' if x < -0.1 else \
#         ('rgba(184, 247, 212, 1)' if x > 0.1 else 'rgba(131, 90, 241, 1)'))
#     dfm = fd[['Word', 'Frequency']]
#     dfm.rename(columns={'Word':'word', 'Frequency':'frequency'}, inplace=True)
    
#     import cv2
#     image = cv2.imread(r"C:\Users\Adam\Desktop\facebook_img.png", 1)
 
#     # plot_wordcloud(data=dfm, backgrounf_color).save(img, format='PNG')
#     wc = WordCloud(background_color = 'white', colormap=plt.get_cmap('PuBuGn'), width=1000, max_words=10, height=1000, random_state=1).generate(content)
#     plt.imshow(wc, interpolation='bilinear')
#     plt.axis("off")

#     image = BytesIO()
#     plt.savefig(image, format='png')
#     image.seek(0)  # rewind the data
#     string = base64.b64encode(image.read())

#     image_64 = 'data:image/png;base64,' + urllib.parse.quote(string)
#     return image_64






if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run_server(debug=False, host='0.0.0.0', port=port)