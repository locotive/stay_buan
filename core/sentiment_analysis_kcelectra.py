from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
from typing import Tuple
import os
import traceback

logger = logging.getLogger(__name__)

class KcELECTRASentimentAnalyzer:
    """KcELECTRA 기반 감성 분석"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or "data/models/kcelectra"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # 모델 파일 확인
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
            
            # 토크나이저와 모델 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                max_length=512,
                truncation=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                num_labels=3
            ).to(self.device)
            self.model.eval()
            
            logger.info("KcELECTRA 모델 초기화 완료")
            
        except Exception as e:
            logger.error(f"KcELECTRA 모델 초기화 중 오류 발생: {str(e)}")
            raise
    
    def predict(self, text: str) -> Tuple[int, float]:
        """텍스트의 감성을 예측"""
        try:
            # 입력 텍스트 전처리
            text = str(text).strip()
            if not text:
                logger.warning("빈 텍스트가 입력되었습니다.")
                return 1, 0.0  # 중립, 낮은 신뢰도
            
            # 토큰화 및 입력 준비
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=True
            ).to(self.device)
            
            # 모델 추론
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                pred = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][pred].item()
            
            return pred, confidence
            
        except Exception as e:
            logger.error(f"감성 분석 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            return 1, 0.0  # 오류 발생 시 중립으로 처리 