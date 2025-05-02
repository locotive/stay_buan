from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class KoBERTSentimentAnalyzer:
    """단일 KoBERT 기반 감성 분석기 (3클래스) - 싱글톤 패턴"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KoBERTSentimentAnalyzer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not KoBERTSentimentAnalyzer._initialized:
            self.model_name = "taeminlee/korean-sentiment-kobert"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            KoBERTSentimentAnalyzer._initialized = True
    
    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment = torch.argmax(probs, dim=1).item()
        confidence = probs.max().item()
        return sentiment, confidence  # 0: 부정, 1: 중립, 2: 긍정
