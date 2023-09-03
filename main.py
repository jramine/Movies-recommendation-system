import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('stopwords')

st.markdown("<h1 style='text-align: center; color: black;'>Movies Recommendation system</h1>", unsafe_allow_html=True)


df1 = pd.read_csv('tmdb_5000_credits.csv')
df2 = pd.read_csv('tmdb_5000_movies.csv')
df1.columns = ['id','tittle','cast','crew']
data= df2.merge(df1,on='id')

stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if not word in stop_words]
    return ' '.join(tokens)

tfidf = TfidfVectorizer(stop_words='english')
data['overview'] = data['overview'].fillna('')
overview = data['overview'].apply(preprocess_text)
title = data['title'].apply(preprocess_text)
tfidf_matrix = tfidf.fit_transform(overview+title)

user_input = st.text_input("Enter a movie you've watched")
if user_input:
    user_input = preprocess_text(user_input)
    user_input = tfidf.transform([user_input])
    cosine_sim = cosine_similarity(user_input, tfidf_matrix)
    top_n = 5  # Number of recommendations to show
    top_n = 5  # Number of recommendations to show
    similar_movies_indices = cosine_sim.argsort()[0][::-1][:top_n]
    similar_movies_data = data.iloc[similar_movies_indices]
    st.table(similar_movies_data[['title', 'overview', 'vote_average']])