import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as en_stop_words
import nltk
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class LemStemTokenizer:
    def __init__(self):
        self.lma = WordNetLemmatizer()
        self.stm = PorterStemmer()
        
    def __call__(self, text):
        text = re.sub(r"[^A-Za-z0-9\-]", " ", text).lower()
        return [self.lma.lemmatize(t) for t in word_tokenize(text)]

class CountVectors:
    def __init__(self, *args, stop_words=True):
        self.tokenizer = LemStemTokenizer()
        self.stop_words = [self.tokenizer(x)[0] for x in en_stop_words] if stop_words else []
        self.vectorizer = CountVectorizer(stop_words=self.stop_words, tokenizer=self.tokenizer)
        self.vectors = self.vectorizer.fit_transform(args)
        self.vectorizer.vocabulary = self.vectorizer.get_feature_names()
        
    def query(self, *args):
        return self.vectorizer.fit_transform(args)
    
    def token_idxs(self, text):
        tokens = self.tokenizer(text)
        idxs = []
        
        for t in tokens:
            try:
                idxs.append(self.vectorizer.vocabulary.index(t))
            except ValueError:
                pass
        return idxs
    
class TFIDFVectors:
    def __init__(self, *args, stop_words=True):
        self.tokenizer = LemStemTokenizer()
        self.stop_words = [self.tokenizer(x)[0] for x in en_stop_words] if stop_words else []
        self.vectorizer = TfidfVectorizer(stop_words=self.stop_words, tokenizer=self.tokenizer)
        self.vectors = self.vectorizer.fit_transform(args)
        self.vectorizer.vocabulary = self.vectorizer.get_feature_names()
        
    def query(self, *args):
        return self.vectorizer.fit_transform(args)
    
    def token_idxs(self, text):
        tokens = self.tokenizer(text)
        idxs = []
        
        for t in tokens:
            try:
                idxs.append(self.vectorizer.vocabulary.index(t))
            except ValueError:
                pass
        return idxs