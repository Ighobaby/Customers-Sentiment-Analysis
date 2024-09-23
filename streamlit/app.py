import streamlit as st
import joblib
from preprocess import clean_text  # Import the preprocessing function

# Load the model and vectorizer from the file
model, vectorizer = joblib.load('best_nb_model.pkl')

# Custom header with white bold text
st.markdown("""
    <style>
    .header {
        font-size: 30px;
        font-weight: bold;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        padding: 20px;
        background-color: #8d0801;
    }
    .centered-image {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
    }
    .social-icons img {
        width: 30px;
        height: auto;
        margin: 0 10px;
        cursor: pointer;
    }
    </style>
    <div class="header">
        Sentiment Analysis on Product Reviews 
        <br>
        by 
        <br>
        Osagie Augustine
    </div>
    """, unsafe_allow_html=True)

# Add and center the image
st.image("images/growth.png", width=700)

review = st.text_area("Enter a product review:")

if st.button("Predict Sentiment"):
    if review:
        with st.spinner('Processing...'):
            # Preprocess the input review
            preprocessed_review = clean_text(review)
            
            # Vectorize the preprocessed review
            vectorized_review = vectorizer.transform([preprocessed_review])
            
            # Predict the sentiment
            prediction = model.predict(vectorized_review)
            
            # Display the result
            if prediction == 1:
                st.success("The review is positive!")
            else:
                st.error("The review is negative!")
    else:
        st.error("Please enter a review.")

# Custom footer with social media links
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #4CAF50;
        color: white;
        text-align: center;
        padding: 10px;
    }
    .footer a {
        color: white;
        text-decoration: none;
        margin: 0 10px;
    }
    </style>
    <div class="footer">
        Developed by Osagie Augustine | ©2024
        <br>
        <a href="https://x.com/Ighobaby" target="_blank">Twitter</a>
        <a href="https://www.linkedin.com/in/augustine-osagie-" target="_blank">LinkedIn</a>
        <a href="https://github.com/Ighobaby" target="_blank">GitHub</a>
    </div>
    """, unsafe_allow_html=True)
