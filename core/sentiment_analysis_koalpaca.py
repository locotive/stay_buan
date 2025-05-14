import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
from typing import Optional, Tuple
import os

logger = logging.getLogger(__name__)

class KoAlpacaSentimentAnalyzer:
    """KoAlpaca 기반 감성 분석기 (8비트 양자화 적용)"""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(KoAlpacaSentimentAnalyzer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, model_path="data/models/koalpaca"):
        if hasattr(self, 'initialized'):
            return
            
        try:
            if not (os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "pytorch_model.bin"))):
                from huggingface_hub import snapshot_download
                logger.info("모델 파일이 없어 자동 다운로드를 시도합니다.")
                snapshot_download(repo_id="beomi/KoAlpaca-Polyglot-12.8B", local_dir=model_path, local_dir_use_symlinks=False)
            
            # 디바이스 설정
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"KoAlpaca 모델 디바이스: {self.device}")
            
            # 토크나이저와 모델 로드
            logger.info(f"KoAlpaca 모델 로드 중... (경로: {model_path})")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=3
            ).to(self.device)
            
            self.model.eval()  # 평가 모드로 설정
            logger.info("KoAlpaca 모델 로드 완료")
            
            self.initialized = True
            
        except Exception as e:
            logger.error(f"KoAlpaca 모델 초기화 중 오류 발생: {str(e)}")
            raise
    
    def predict(self, text: str) -> Tuple[int, float]:
        """텍스트의 감성 분석 수행"""
        try:
            # 토크나이징
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # 추론
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                
            # 결과 처리
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0][pred].item()
            
            return pred, conf
            
        except Exception as e:
            logger.error(f"KoAlpaca 감성 분석 중 오류 발생: {str(e)}")
            return 1, 0.0  # 오류 시 중립으로 처리 