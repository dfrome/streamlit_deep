import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import Normalizer

st.title("Modèle Word2Vec")

# Load the pre-trained model
model = load_model('word2vec.h5')
vectors = model.layers[0].get_weights()[0]

# Define a sample vocabulary (this should match the tokenizer used during training)
sample_vocabulary = {
    'example': 1,
    'word': 2,
    'test': 3,
    'sample': 4,
    'language': 5,
    'model': 6,
    'vector': 7,
    'embedding': 8,
    'neural': 9,
    'network': 10
}

# Reverse the vocabulary mapping
idx2word = {v: k for k, v in sample_vocabulary.items()}

# Function to calculate cosine similarity
def dot_product(vec1, vec2):
    return np.sum((vec1 * vec2))

def cosine_similarity(vec1, vec2):
    return dot_product(vec1, vec2) / np.sqrt(dot_product(vec1, vec1) * dot_product(vec2, vec2))

# Function to find closest words
def find_closest(word_index, vectors, number_closest=10):
    list1 = []
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist, index])
    return np.asarray(sorted(list1, reverse=True)[:number_closest])

# Function to return closest words
def print_closest(word, vectors, number=10):
    index_closest_words = find_closest(sample_vocabulary[word], vectors, number)
    closest_words = [(idx2word[index_word[1]], index_word[0]) for index_word in index_closest_words]
    return closest_words

# Widget to input a word
word = st.text_input("Entrez un mot:")

if word:
    if word in sample_vocabulary:
        closest_words = print_closest(word, vectors)
        st.write(f"Les 10 mots les plus proches de '{word}' sont :")
        for word, similarity in closest_words:
            st.write(f"{word} -- {similarity:.4f}")
    else:
        st.error("Le mot n'est pas dans le vocabulaire.")
