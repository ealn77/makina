import streamlit as st
import pickle
from nltk.tokenize import word_tokenize

with open("logistic_model.pcl", "rb") as f:
    model = pickle.load(f)
with open("tfidf_vectorizer.pcl", "rb") as f:
    vectorizer = pickle.load(f)

def temizle(kelimeler):
    return [k.lower() for k in kelimeler if k.isalpha()]

st.title("🎬 Film Yorumu Duygu Analizi")
yorum = st.text_area("Yorumunuzu yazın:")

if st.button("Tahmin Et"):
    tokens = temizle(word_tokenize(yorum))
    veri = vectorizer.transform([" ".join(tokens)])
    tahmin = model.predict(veri)[0]
    st.success("🔍 Tahmin: " + ("Olumlu" if tahmin == 1 else "Olumsuz"))
