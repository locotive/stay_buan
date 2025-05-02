from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

class SentimentAnalyzer:
    """감성 분석기 (싱글톤 패턴)"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SentimentAnalyzer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not SentimentAnalyzer._initialized:
            try:
                self.okt = Okt()
                self.vectorizer = CountVectorizer()
                self.model = MultinomialNB()
                SentimentAnalyzer._initialized = True
            except OSError as e:
                if "JVM is already started" in str(e):
                    # JVM이 이미 시작된 경우, 오류를 무시하고 진행
                    SentimentAnalyzer._initialized = True
                else:
                    # 다른 OSError는 그대로 발생
                    raise
    
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