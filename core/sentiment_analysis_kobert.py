from transformers import BertTokenizer, BertForSequenceClassification
import torch

class KoBERTSentimentAnalyzer:
    """KoBERT 기반 감성 분석기"""
    
    def __init__(self, model_path='path/to/fine-tuned-model'):
        self.tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
        self.model = BertForSequenceClassification.from_pretrained(model_path)
    
    def predict(self, text):
        """텍스트 감성 예측"""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment = torch.argmax(probs, dim=1).item()
        confidence = probs.max().item()
        return sentiment, confidence  # 0: 부정, 1: 중립, 2: 긍정 