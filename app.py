import streamlit as st
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================================
# Page Configuration
# ==========================================================
st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="🧠",
    layout="centered"
)

# ==========================================================
# Load Resources
# ==========================================================
@st.cache_resource
def load_resources():
    model = load_model("lstm_model.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len


@st.cache_data
def load_dataset():
    return pd.read_csv("qoute_dataset.csv")


model, tokenizer, max_len = load_resources()
quotes_df = load_dataset()

# ==========================================================
# Prediction Function
# ==========================================================
def predict_next_word(text):

    sequence = tokenizer.texts_to_sequences([text])[0]

    if len(sequence) == 0:
        return "No prediction available."

    sequence = pad_sequences(
        [sequence],
        maxlen=max_len - 1,
        padding="pre"
    )

    prediction = model.predict(sequence, verbose=0)

    predicted_index = np.argmax(prediction)

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word

    return "No prediction available."


# ==========================================================
# Header
# ==========================================================
st.title("🧠 Next Word Prediction")

st.markdown("### Developed by **Aniket Sharma**")

st.write(
    """
This web application predicts the **next word** in a sentence using a
Long Short-Term Memory (**LSTM**) neural network trained on a custom quote dataset.
"""
)

# ==========================================================
# User Input
# ==========================================================
user_input = st.text_input(
    "Enter a sentence",
    placeholder="Example: Success comes to those who"
)

if st.button("Predict Next Word", use_container_width=True):

    if user_input.strip() == "":
        st.warning("Please enter a sentence.")

    else:

        predicted_word = predict_next_word(user_input)

        st.success(f"### Predicted Next Word: **{predicted_word}**")

# ==========================================================
# Disclaimer
# ==========================================================
st.markdown("---")

st.warning(f"""
### ⚠ Disclaimer

This model **has not been trained on a large-scale language dataset** like modern Large Language Models (LLMs).

Instead, it has been trained only on the provided **Quote Dataset (`qoute_dataset.csv`)** containing:

- **Total Quotes (Rows): {quotes_df.shape[0]:,}**
- **Total Columns: {quotes_df.shape[1]}**

Because the model has learned only from these **{quotes_df.shape[0]:,} quotes**, it performs well mainly on sentence patterns similar to the training data.

It is **not intended for general real-world next-word prediction**, and predictions on unseen sentence structures may be inaccurate.

You can explore the complete training dataset in the section below.
""")

# ==========================================================
# Training Dataset
# ==========================================================
with st.expander("📖 See More - Training Quotes"):

    st.write(
        f"The model was trained using the following **{quotes_df.shape[0]:,} quotes**."
    )

    st.dataframe(
        quotes_df,
        use_container_width=True,
        hide_index=True
    )

# ==========================================================
# Footer
# ==========================================================
st.markdown("---")

st.markdown(
    "<center><b>Next Word Prediction</b><br>"
    "Developed by Aniket Sharma<br>"
    "TensorFlow • Keras • Streamlit</center>",
    unsafe_allow_html=True
)
