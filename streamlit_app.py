import streamlit as st
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D

st.title("Modèle Word2Vec")

embedding_dim = 300
vocab_size=10000
WINDOW_SIZE=5

#model = Sequential()
#model.add(Embedding(vocab_size, embedding_dim, input_length=WINDOW_SIZE))
#model.add(GlobalAveragePooling1D())
#model.add(Dense(vocab_size, activation='softmax'))
#
#model.load_weights("word2vec.h5")
#print("Poids du modèle chargés.")

# alternative
# Charger le modèle entier
from tensorflow.keras.models import load_model
model = load_model('word2vec_model.h5')
print("Modèle chargé avec succès!")

# Extraire la matrice d'embeddings
embedding_matrix = model.layers[0].get_weights()[0]
print("Matrice d'embeddings extraite avec succès!")
# end of alternative

vectors = model.layers[0].trainable_weights[0].numpy()
import numpy as np
from sklearn.preprocessing import Normalizer

def dot_product(vec1, vec2):
    return np.sum((vec1*vec2))

def cosine_similarity(vec1, vec2):
    return dot_product(vec1, vec2)/np.sqrt(dot_product(vec1, vec1)*dot_product(vec2, vec2))

def find_closest(word_index, vectors, number_closest):
    list1=[]
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist,index])
    return np.asarray(sorted(list1,reverse=True)[:number_closest])

def compare(index_word1, index_word2, index_word3, vectors, number_closest):
    list1=[]
    query_vector = vectors[index_word1] - vectors[index_word2] + vectors[index_word3]
    normalizer = Normalizer()
    query_vector =  normalizer.fit_transform([query_vector], 'l2')
    query_vector= query_vector[0]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist,index])
    return np.asarray(sorted(list1,reverse=True)[:number_closest])

def print_closest(word, number=5):
    index_closest_words = find_closest(word2idx[word], vectors, number)
    for index_word in index_closest_words :
        print(idx2word[index_word[1]]," -- ",index_word[0])

# Widget pour demander un mot à l'utilisateur
word = st.text_input("Entrez un mot:")
if word:
    print_closest(word)