import os
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from faqs import faqs

# prepare nltk folder
nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except:
        nltk.download('punkt', download_dir=nltk_data_path)

    try:
        nltk.data.find('corpora/stopwords')
    except:
        nltk.download('stopwords', download_dir=nltk_data_path)

setup_nltk()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in string.punctuation]
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    return " ".join(tokens)

# Build model once
questions = [preprocess(q) for q, a in faqs]
answers = [a for q, a in faqs]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

def get_response(user_input):
    user_input = preprocess(user_input)
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    idx = similarity.argmax()

    if similarity[0][idx] < 0.3:
        return "Sorry, I didn't understand that."

    return answers[idx]
