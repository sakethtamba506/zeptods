import streamlit as st
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

st.title('Input and Output Box Example')

user_input = st.text_input("Type something here:")

data = joblib.load("data.joblib")

tfidf_matrix = vectorizer.fit_transform(data['searchable_text'])


class RealTimeSearchSystem:
    def __init__(self, tfidf_matrix, data, vectorizer):
        self.tfidf_matrix = tfidf_matrix
        self.data = data
        self.vectorizer = vectorizer

    def search(self, query, top_n=10):
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_n:][::-1]
        return self.data.iloc[top_indices][['product_name', 'brand', 'product_category_tree', 'description']], similarities[top_indices]
    
search_system = RealTimeSearchSystem(tfidf_matrix, data, vectorizer)

if user_input:
    search_results, similarities = search_system.search(user_input)
    st.write(search_results)
else:
    st.write("Please enter some text.")
