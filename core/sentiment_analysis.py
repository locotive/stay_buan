from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

class SentimentAnalyzer:
    """감성 분석기"""
    
    def __init__(self):
        self.okt = Okt()
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()
    
    def preprocess(self, text):
        """텍스트 전처리 및 형태소 분석"""
        tokens = self.okt.morphs(text, stem=True)
        return ' '.join(tokens)
    
    def train(self, data, labels):
        """모델 학습"""
        processed_data = [self.preprocess(text) for text in data]
        X = self.vectorizer.fit_transform(processed_data)
        self.model.fit(X, labels)
    
    def predict(self, text):
        """감성 예측"""
        processed_text = self.preprocess(text)
        X = self.vectorizer.transform([processed_text])
        return self.model.predict(X) 