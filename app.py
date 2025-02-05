import tensorflow as tf
import json
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from deep_translator import GoogleTranslator
import streamlit as st

# Load the model
model = tf.keras.models.load_model('C:/Users/Wasseem/Desktop/project/language detection/app/best_model_language_detection_v10 (1).keras')
label_to_num = pickle.load(open('saved/label_to_num.pkl','rb'))
Token = pickle.load(open('saved/Token.pkl','rb'))
num_to_label = pickle.load(open('saved/num_to_label.pkl','rb'))

# Load the JSON file containing language codes and names
def load_language_json(file_path):
    with open(file_path, 'r') as f:
        language_dict = json.load(f)
    return language_dict

language_dict = load_language_json('C:/Users\Wasseem/Desktop/project/language detection/app/lan_to_language.json')

def predict_language(sentence, model, tokenizer, maxlen=140, num_to_label=num_to_label):
    # Step 1: Tokenize and pad the input sentence
    seq = tokenizer.texts_to_sequences([sentence])
    padded_seq = pad_sequences(seq, padding='post', maxlen=maxlen, truncating='post')

    # Step 2: Predict the language (get the index of the class with highest probability)
    prediction = model.predict(padded_seq)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Step 3: Map the predicted class to the language label
    predicted_language = num_to_label[predicted_class]

    return predicted_language

def translate_text(text, language_name):
    """Translate text into the specified language using its name."""
    try:
        translated_text = GoogleTranslator(source="auto", target=language_name.lower()).translate(text)
        return translated_text
    except Exception as e:
        return f"Error: {str(e)}"



st.title("This is a language detection using LSTM")

text = st.text_area("Enter text", height=100)

option = st.radio("Choose an option:", ["Detect Language", "Translate"])

if option == "Detect Language":
    if st.button("Detect"):
        if text.strip():
            detected_language = predict_language(text, model, Token)
            st.success(f"Detected Language: {language_dict[detected_language]}")
        else:
            st.warning("Please enter some text.")

# If user selects "Translate"
elif option == "Translate":
    target_lang = st.text_input("Choose language to translate to (e.g., 'french', 'german')")
    
    if st.button("Translate"):
        if text.strip() and target_lang.strip():
            try:
                translated_text = GoogleTranslator(source="auto", target=target_lang.lower()).translate(text)
                st.text_area("Result:", translated_text, height=100)
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter text and target language.")