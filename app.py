import nltk
nltk.download("punkt")
# app.py

import streamlit as st
import pickle
from nltk.tokenize import word_tokenize

# Model ve vektörizer'ı yükle
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Temizleme fonksiyonu
def temizle(kelimeler):
    return [k.lower() for k in kelimeler if k.isalpha()]

# Streamlit arayüzü
st.title("🎬 Film Yorumu Duygu Analizi")
yorum = st.text_area("Yorumunuzu yazın:")

if st.button("Tahmin Et"):
    if yorum.strip() == "":
        st.warning("Lütfen bir yorum girin.")
    else:
        tokens = temizle(word_tokenize(yorum))
        veri = vectorizer.transform([" ".join(tokens)])
        tahmin = model.predict(veri)[0]
        st.success("🔍 Tahmin: " + ("Olumlu" if tahmin == 1 else "Olumsuz"))
