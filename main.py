import datetime as dt
import re
import pandas as pd
import streamlit as st
from flair.data import Sentence
from flair.models import TextClassifier
from twitterscraper import query_tweets

st.title('Twitter Sentiment Analysis')
st.subheader("An ML app to detect the sentiments behind the tweets using Flair -a state-of-art NLP library and a query finder that crawls for specified tweets")

with st.spinner("Loading Classification Model....."):
    classifier = TextClassifier.load('model-saves/final-model.pt')

allowed_chars = ' AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789~`!@#$%^&*()-=_+[]{}|;:",./<>?'
punct = '!?,.@#'
maxlen = 280

def preprocess(text):
    return ''.join([' ' + char + ' ' if char in punct else char for char in [char for char in re.sub(r'http\S+', 'http', text, flags=re.MULTILINE) if char in allowed_chars]])[:maxlen]

st.subheader("Single tweet classification")

tweet_input = st.text_input('PUT YOUR TWEET :- ')

if tweet_input != '':
    sentence = Sentence(preprocess(tweet_input))

    with st.spinner('Predicting...'):
        classifier.predict(sentence)

    label_dict = {'0': 'Negative', '4':'Positive'}

    if len(sentence.labels) > 0:
        st.write('Prediction: ')
        st.write(label_dict[sentence.labels[0].value] + ' with',
                 sentence.labels[0].score*100, '% confidence')


st.subheader("Search Twitter for Query")

query = st.text_input("Query:", '#')


if query != '' and query != '#':
    with st.spinner(f'Searching for and analysing {query}...'):
        tweets = query_tweets(query, begindate = dt.date.today() - dt.timedelta(weeks=4), lang='en')

        tweet_data = pd.DataFrame({
            'tweet':[],
            'predicted-sentiment':[]
        })

        pos_vs_neg = {'0':0,'4':0}

        for tweet in tweets:
            if tweet.text in ('',' '):
                continue
            sentence = Sentence(preprocess(tweet.text))
            classifier.predict(sentence)
            sentiment = sentence.labels[0]
            pos_vs_neg[sentiment.value] += 1
            tweet_data = tweet_data.append({'tweet':tweet.text,'predicted-sentiment':sentiment}, ignore_index=True)




try:
    st.write(tweet_data)
    try:
        st.write('Positive to negative tweet ratio:',pos_vs_neg['4']/pos_vs_neg['0'])
    except ZeroDivisionError:
        st.write('All Positive tweets')
except NameError:
    pass
