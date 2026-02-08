import numpy as np
import pandas as pd
import streamlit as st 
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


# Load the Imdb Dataset word Index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}


## Load the pre-trained model with Relu Activation  
model = load_model('simple_rnn_imdb.h5')


## Step 2: Helper Functions 
# Function to Decode Reviews 
def decode_review(encoded_review):
  return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])


# Function to Preprocessor User Input 
def preprocess_text(text):
  words = text.lower().split()
  encoded_review = [word_index.get(word, 2) + 3 for word in words]
  padded_review = sequence.pad_sequences([encoded_review], maxlen=400)
  return padded_review


## Streamlit App 
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a Movie Review to Classify Whether The Movie Review is a Positive or Negative")

# User Input 
user_input = st.text_area('Movie Review')

if st.button('Classifify'):
  preprocessed_input = preprocess_text(user_input)
  prediction  = model.predict(preprocessed_input)

  sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

  st.write(f"Sentiment Analysis: {sentiment}")
  st.write(f"Prediction Score: {prediction[0][0]}")
else:
  
  st.write('Please Enter a Movie Review.')