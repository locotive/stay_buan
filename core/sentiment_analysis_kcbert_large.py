from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
from typing import Tuple
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class KcBERTLargeSentimentAnalyzer:
    """KcBERT-large 기반 감성 분석 모델"""
    
    def __init__(self, model_path: str = None):
        """
        KcBERT-large 모델 초기화
        
        Args:
            model_path (str, optional): 로컬 모델 경로. None인 경우 기본 경로 사용
        """
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 모델 경로 설정 및 확인
            if model_path is None:
                model_path = str(Path("data/models/kcbert-large").absolute())
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"모델 디렉토리를 찾을 수 없습니다: {model_path}")
            
            # 필수 파일 존재 확인
            required_files = ["config.json", "pytorch_model.bin", "vocab.txt"]
            for file in required_files:
                file_path = os.path.join(model_path, file)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"필수 파일을 찾을 수 없습니다: {file_path}")
            
            logger.info(f"KcBERT-large 모델 파일 확인 완료: {model_path}")
            
            # 토크나이저와 모델 로드 (로컬 파일 강제 사용)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True,
                use_fast=True,
                max_length=300,
                truncation=True
            )
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=3,
                local_files_only=True,
                trust_remote_code=True
            ).to(self.device)
            
            self.model.eval()
            logger.info(f"KcBERT-large 모델 초기화 완료 (device: {self.device}, path: {model_path})")
            
        except Exception as e:
            logger.error(f"KcBERT-large 모델 초기화 중 오류 발생: {str(e)}")
            raise
    
    def predict(self, text: str) -> Tuple[int, float]:
        """
        텍스트의 감성 분석 수행
        
        Args:
            text (str): 분석할 텍스트
            
        Returns:
            tuple: (감성 레이블 정수, 신뢰도 점수)
        """
        try:
            if not text or not text.strip():
                logger.warning("빈 텍스트가 입력되었습니다")
                return 1, 0.0  # 중립으로 처리
            
            # 토큰화 및 모델 입력 준비
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=300,
                padding="max_length"
            ).to(self.device)
            
            # 추론
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                pred_label = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][pred_label].item()
            
            logger.debug(f"KcBERT-large 예측 결과: {pred_label} (신뢰도: {confidence:.3f})")
            return pred_label, confidence
            
        except Exception as e:
            logger.error(f"KcBERT-large 예측 중 오류 발생: {str(e)}")
            return 1, 0.0  # 오류 시 중립으로 처리 