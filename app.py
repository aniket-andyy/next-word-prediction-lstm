import streamlit as st
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="🧠",
    layout="centered"
)

# =====================================================
# Load Resources
# =====================================================
@st.cache_resource
def load_resources():
    model = load_model("lstm_model (1).h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len


@st.cache_data
def load_quotes():
    return pd.read_csv("qoute_dataset.csv")


model, tokenizer, max_len = load_resources()
quotes_df = load_quotes()

# =====================================================
# Prediction Function
# =====================================================
def predict_next_word(text):

    sequence = tokenizer.texts_to_sequences([text])[0]

    if len(sequence) == 0:
        return "Unknown"

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

    return "Unknown"


# =====================================================
# Header
# =====================================================
st.title("🧠 Next Word Prediction")

st.subheader("Developed by Aniket Sharma")

st.write(
    """
This application uses an **LSTM (Long Short-Term Memory)** neural network
to predict the next word based on the text entered by the user.
"""
)

# =====================================================
# User Input
# =====================================================
user_input = st.text_input(
    "Enter a sentence",
    placeholder="Example: Success comes to those who"
)

if st.button("🔮 Predict Next Word", use_container_width=True):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:

        predicted_word = predict_next_word(user_input)

        st.success(
            f"### Predicted Next Word: **{predicted_word}**"
        )

# =====================================================
# Disclaimer
# =====================================================
st.markdown("---")

st.warning(f"""
## ⚠ Disclaimer

This model has **not been trained on a large-scale language dataset** like modern AI language models.

It has been trained **only on the Quote Dataset (`qoute_dataset.csv`)**, which contains:

- **Total Rows (Quotes): {quotes_df.shape[0]:,}**
- **Total Columns: {quotes_df.shape[1]}**

Since the training dataset consists of only **{quotes_df.shape[0]:,} quotes**, the model has learned patterns only from these sentences.

Because of this limited and domain-specific training data, the model **is not intended for general real-world next-word prediction** and may produce inaccurate or unexpected predictions for sentences outside the training dataset.

To understand what the model has learned, you can explore all the training quotes in the **"See More Quotes Used for Training"** section below.
""")

# =====================================================
# Show Training Quotes
# =====================================================
with st.expander("📖 See More Quotes Used for Training"):

    st.write(
        f"""
These are the **{quotes_df.shape[0]:,} quotes**
used to train the LSTM model.
"""
    )

    st.dataframe(
        quotes_df,
        use_container_width=True,
        hide_index=True
    )

# =====================================================
# Footer
# =====================================================
st.markdown("---")

st.caption(
    "🧠 Next Word Prediction | Developed by Aniket Sharma"
)