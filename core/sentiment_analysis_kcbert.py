import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Transformers 오프라인 모드 설정
os.environ["TRANSFORMERS_OFFLINE"] = "1"

class KCBERTSentimentAnalyzer:
    """단일 KCBERT 기반 감성 분석기 (3클래스) - 싱글톤 패턴"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KCBERTSentimentAnalyzer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not KCBERTSentimentAnalyzer._initialized:
            self.model_name = "beomi/kcbert-base"
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, local_files_only=True)
                KCBERTSentimentAnalyzer._initialized = True
            except Exception as e:
                print("🔴 감성 분석 모델 로딩 실패:", e)
                print("⚠️ Hugging Face 모델 캐시가 없거나 인터넷이 필요합니다.")
                self.model = None
                self.tokenizer = None
    
    def predict(self, text):
        if self.model is None or self.tokenizer is None:
            print("⚠️ 감성 분석 모델이 로드되지 않아 분석을 건너뜁니다.")
            return 1, 0.0  # 중립(1) 반환, 신뢰도 0
            
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment = torch.argmax(probs, dim=1).item()
        confidence = probs.max().item()
        return sentiment, confidence  # 0: 부정, 1: 중립, 2: 긍정 