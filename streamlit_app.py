"""
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import Normalizer

st.title("Modèle Word2Vec")

# Load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model_and_vectors():
    model = load_model('word2vec.h5')
    vectors = model.layers[0].get_weights()[0]
    return model, vectors

model, vectors = load_model_and_vectors()

# Display model summary
with st.expander("Model Summary"):
    st.write(model.summary())

# Example of using word vectors
def dot_product(vec1, vec2):
    return np.sum((vec1 * vec2))

def cosine_similarity(vec1, vec2):
    return dot_product(vec1, vec2) / np.sqrt(dot_product(vec1, vec1) * dot_product(vec2, vec2))

# Function to find closest words
def find_closest(word_index, vectors, number_closest=5):
    list1 = []
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist, index])
    return np.asarray(sorted(list1, reverse=True)[:number_closest])

# User inputs for word similarity
st.sidebar.header("Word Similarity Inputs")
word_index_1 = st.sidebar.number_input("Enter first word index:", min_value=0, max_value=len(vectors)-1, value=0)
word_index_2 = st.sidebar.number_input("Enter second word index:", min_value=0, max_value=len(vectors)-1, value=1)

# Calculate and display cosine similarity
vector_1 = vectors[word_index_1]
vector_2 = vectors[word_index_2]
similarity = cosine_similarity(vector_1, vector_2)
st.sidebar.write(f'Cosine similarity between word {word_index_1} and word {word_index_2}: {similarity:.4f}')

# User inputs for finding closest words
st.header("Find Closest Words")
word_index = st.number_input("Enter word index to find closest words:", min_value=0, max_value=len(vectors)-1, value=0)
number_closest = st.number_input("Number of closest words:", min_value=1, max_value=10, value=5)
closest_words = find_closest(word_index, vectors, number_closest)

# Display closest words
st.write(f"Closest words to word index {word_index}:")
for dist, idx in closest_words:
    st.write(f"Word index: {idx}, Cosine similarity: {dist:.4f}")

"""
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import Normalizer

st.title("Modèle Word2Vec")

# Define the file path for the model
file_path = 'word2vec.h5'

# Load the pre-trained model
try:
    model = load_model(file_path)
    vectors = model.layers[0].get_weights()[0]

    # Display model summary
    with st.expander("Model Summary"):
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        st.text('\n'.join(model_summary))

    # Example of using word vectors
    def dot_product(vec1, vec2):
        return np.sum((vec1 * vec2))

    def cosine_similarity(vec1, vec2):
        return dot_product(vec1, vec2) / np.sqrt(dot_product(vec1, vec1) * dot_product(vec2, vec2))

    # Function to find closest words
    def find_closest(word_index, vectors, number_closest=5):
        list1 = []
        query_vector = vectors[word_index]
        for index, vector in enumerate(vectors):
            if not np.array_equal(vector, query_vector):
                dist = cosine_similarity(vector, query_vector)
                list1.append([dist, index])
        return np.asarray(sorted(list1, reverse=True)[:number_closest])

    # User inputs for word similarity
    st.sidebar.header("Word Similarity Inputs")
    word_index_1 = st.sidebar.number_input("Enter first word index:", min_value=0, max_value=len(vectors)-1, value=0)
    word_index_2 = st.sidebar.number_input("Enter second word index:", min_value=0, max_value=len(vectors)-1, value=1)

    # Calculate and display cosine similarity
    vector_1 = vectors[word_index_1]
    vector_2 = vectors[word_index_2]
    similarity = cosine_similarity(vector_1, vector_2)
    st.sidebar.write(f'Cosine similarity between word {word_index_1} and word {word_index_2}: {similarity:.4f}')

    # User inputs for finding closest words
    st.header("Find Closest Words")
    word_index = st.number_input("Enter word index to find closest words:", min_value=0, max_value=len(vectors)-1, value=0)
    number_closest = st.number_input("Number of closest words:", min_value=1, max_value=10, value=5)
    closest_words = find_closest(word_index, vectors, number_closest)

    # Display closest words
    st.write(f"Closest words to word index {word_index}:")
    for dist, idx in closest_words:
        st.write(f"Word index: {idx}, Cosine similarity: {dist:.4f}")

except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")




"""
import streamlit as st
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
import tensorflow as tf
from tensorflow.keras.models import load_model
st.title("Modèle Word2Vec")
# Load the pre-trained model
model = load_model('word2vec.h5')

# Extract word vectors from the embedding layer
#vectors = model.layers[0].trainable_weights[0].numpy()

# Define the vocabulary size
vocab_size = 10000  # Example value, replace with your actual vocabulary size
embedding_dim = 300
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(GlobalAveragePooling1D())
model.add(Dense(vocab_size, activation='softmax'))

#model.load_weights("word2vec.h5")
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

def print_closest(word, number=10):
    index_closest_words = find_closest(word2idx[word], vectors, number)
    for index_word in index_closest_words :
        print(idx2word[index_word[1]]," -- ",index_word[0])

#Exemple d'utilisation de la fonction print_closest
print_closest('zombie')

"""