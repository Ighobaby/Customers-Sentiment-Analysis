# Project Title
Customer-Sentiment-Analysis

# Introduction/Overview
In today's e-commerce landscape, analyzing customer feedback is crucial for businesses to make data-driven decisions. This project focuses on building an end-to-end sentiment analysis solution for customer review for our business stakeholers. By leveraging Data Science and Machine Learning techniques, the project aims to preprocess, analyze, and visualize customer sentiments from textual reviews. A successful sentiment analysis implementation will significantly enhance customer trust and business growth..

# Business Problem
In this project, I used historical customer data to build a predictive model and evaluate its accuracy. 
The model will help the company take proactive actions based on the likelihood of customer churn.This E-commerce platforms receive thousands of reviews daily. Manually analyzing these reviews is inefficient and time-consuming. The goal of this project is to develop an automated sentiment analysis system that can process and interpret these reviews to provide insights into customer satisfaction and areas of improvement.

# Technologies Used
    Programming Language: Python
    Libraries/Frameworks:
        Data Handling: numpy, pandas
        Data Visualization: matplotlib, seaborn
        Natural Language Processing: nltk, re
        Machine Learning: sklearn (including train_test_split, MultinomialNB, LogisticRegression,      RandomForestClassifier, SVC)
        Sentiment Analysis Pre-trained Models: VADER, TextBlob
        Web Framework: Flask, Streamlit
        Model Serialization: pickle
        Feature Extraction: CountVectorizer, TfidfVectorizer
        Word Cloud Generation: wordcloud

# How to Run the Project
1. Clone the repository:
  git clone https://github.com/Ighobaby/customer-sentiment-analysis.git
  cd customer-sentiment-analysis

2. Install the required dependencies:
   pip install -r requirements.txt

3. Run Jupyter Notebooks for Analysis:
    Open the notebooks/ folder.
    Run the following notebooks in order:
        SENTIMENT ANALYSIS EDA, PREPROCESSING & FEATURE ENGINEERING-.ipynb – Loading, EDA, Preprocessing and Feature Extraction
        SENTIMENT ANALYSIS MODEL BUILDING.ipynb – Model Training and Evaluation

4. Run the Flask or Streamlit App:
   For Flask:
     python app.py
  For Streamlit:
    streamlit run app.py

5. Input Review Text: Once the app is running, input a review for sentiment analysis and get instant predictions.


# Key Findings
    The Naive Bayes model achieved an accuracy of 97.47% and an F1 score of 97.20%.
    Comparison with pre-trained models and custom models resulted in the following scores:

Model	                        Accuracy	  F1 Score	  Precision	  Recall
Naive Bayes	                  97.47%	    97.24%	    97.36%	    97.47%
Logistic Regression           97.67%	    97.46%	    97.56%	    97.67%
Random Forest	                99.17%	    99.07%	    99.00%	    99.17%
Support Vector Classifier	    97.64%	    97.43%	    97.52%	    97.64%

# Results and Findings
The Random Forest model performed the best, achieving an accuracy of 99.17%. However, Naive Bayes is preferred for its simplicity and ease of deployment, offering competitive results. The insights gained from sentiment analysis can inform customer service strategies and product development.

# Model Deployment
The model is deployed using Flask and Streamlit. Users can input a review text, and the model will predict whether the sentiment is positive, negative, or neutral. The web application provides an easy interface for real-time sentiment predictions.

# Limitations
    The target variable is highly imbalanced
    The model struggles with interpreting sarcastic or ambiguous statements.
    It has difficulty handling context where sentiment may not be clear from individual words.

# Next Steps
    Improve the model’s ability to handle sarcasm and ambiguous language.
    Integrate additional datasets for broader coverage and improve robustness.
    Continuously update the sentiment lexicon to adapt to changing trends in language.

# Conclusion
  The sentiment analysis system developed successfully processes product reviews. The results from the project will help inform product development and improve customer satisfaction.

# Recommendations

    Explore methods for better handling sarcasm and ambiguous reviews.
    Integrate additional data sources to make the model more generalizable.
    Keep updating the model with new data to capture emerging trends and language use.
