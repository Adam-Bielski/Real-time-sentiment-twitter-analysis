# Real-time-sentiment-twitter-analysis


Link to an application: https://twitter-dashbourd.herokuapp.com/. (You have to wait around ~20 second for the app to start; afterwards refresh the website)

#### Aim

Twitter data can provide us with real-time general opinions on a given topics/brand/company. It's the next generation of survey-based analyses.
This is a perfect way to assess the image of a company, when you run it or want to invest in it.

Based on the sentiment analysis of tweets we can check how many of them use: negative, neutral or positive words, across time
This time we track "Biden" word to know what the world thinks of him! 

#### Execution of the app

Thanks to the Twitter Developer account you can stream real-time tweets. They are continuously saved on the PostgreSQL databases on the Heroku, which apart from storing the data in cloud enables you to run an application written in Python (Dash & Plotly).


#### Files used in app development

1. app.py - Application itself with dashboards and data transformations
2. scraping.py - Script saving stream of twitter data into ProsgreSQL database on Heroku cloud
3. scraping_server.py - additional file helping to run an application
4. runtime.txt - version of the python used
5. requirements.txt - versions of packages used
6. Procfile - file specifying two actions (streaming and updating dashboard)









