import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def clean_text(text):
    text = str(text)
    text = text.lower() # to convert words to lowercase
    text = re.sub(r"[^a-zA-Z\s]", "", text) # remove special characters and numbers
    text = re.sub(r"\st", " ", text).strip() # remove extra spaces
    tokens = word_tokenize(text) # tokenize text
    tokens = [word for word in tokens if word not in stop_words] # remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens] # lemmatize tokens
    return " ".join(tokens)

data['clean_review'] = data['Review'].apply(clean_text)