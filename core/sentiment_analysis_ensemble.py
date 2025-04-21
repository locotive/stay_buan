from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class EnsembleSentimentAnalyzer:
    """3개 감성 분석 모델 앙상블"""
    
    def __init__(self):
        self.models = []

        model_configs = [
            # ✅ 현재는 이 모델이 삭제 상태 — 대체 필요
            # {
            #     'name': 'brainchicken/kobert-sentiment',
            #     'tokenizer': AutoTokenizer.from_pretrained('brainchicken/kobert-sentiment'),
            #     'model': AutoModelForSequenceClassification.from_pretrained('brainchicken/kobert-sentiment')
            # },
            {
                'name': 'nlp04/kobert-news-sentiment',
                'tokenizer': AutoTokenizer.from_pretrained('nlp04/kobert-news-sentiment'),
                'model': AutoModelForSequenceClassification.from_pretrained('nlp04/kobert-news-sentiment')
            },
            {
                'name': 'taeminlee/korean-sentiment-kobert',  # 대체 모델
                'tokenizer': AutoTokenizer.from_pretrained('taeminlee/korean-sentiment-kobert'),
                'model': AutoModelForSequenceClassification.from_pretrained('taeminlee/korean-sentiment-kobert')
            },
            {
                'name': 'beomi/kcbert-base',  # 분류기는 없어서 아래 처리 주의
                'tokenizer': AutoTokenizer.from_pretrained('beomi/kcbert-base'),
                'model': AutoModelForSequenceClassification.from_pretrained('taeminlee/korean-sentiment-kobert')  # ✅ 대신 감성 분류 모델을 붙임
            }
        ]

        for m in model_configs:
            self.models.append({
                'tokenizer': m['tokenizer'],
                'model': m['model']
            })
    
    def predict(self, text):
        avg_probs = torch.zeros(3)

        for m in self.models:
            inputs = m['tokenizer'](text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            outputs = m['model'](**inputs)
            logits = outputs.logits

            if logits.size(-1) == 2:
                probs = torch.nn.functional.softmax(logits, dim=-1)
                probs = torch.cat([probs[0][:1], torch.tensor([0.0]), probs[0][1:]])  # 부정, 중립(0), 긍정
            else:
                probs = torch.nn.functional.softmax(logits, dim=-1)[0]

            avg_probs += probs

        avg_probs /= len(self.models)
        sentiment = torch.argmax(avg_probs).item()
        confidence = avg_probs.max().item()

        return sentiment, confidence
