import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
from typing import Tuple, Optional
import os
import re
import traceback

logger = logging.getLogger(__name__)

class KoBERTSentimentAnalyzer:
    """KoBERT 기반 감성 분석"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or "data/models/kobert"
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
            
            logger.info("KoBERT 모델 초기화 완료")
            
        except Exception as e:
            logger.error(f"KoBERT 모델 초기화 중 오류 발생: {str(e)}")
            raise
    
    def predict(self, text: str) -> Tuple[int, float]:
        """텍스트의 감성을 예측"""
        try:
            # 입력 텍스트 전처리
            text = str(text).strip()
            if not text:
                logger.warning("빈 텍스트가 입력되었습니다.")
                return 1, 0.0  # 중립, 낮은 신뢰도
            
            # 텍스트 길이 검증
            if len(text) > 2048:  # 한글 기준 약 512 토큰
                text = text[:2048]
                logger.warning("텍스트가 2048자를 초과하여 잘립니다.")
            
            # 토큰화 시도
            try:
                # 토큰화 및 입력 준비 (단계별로 수행)
                encoded = self.tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    add_special_tokens=True,
                    return_tensors="pt",
                    return_token_type_ids=True  # 명시적으로 token_type_ids 요청
                )
                
                # 필수 입력 필드 확인 및 생성
                input_ids = encoded['input_ids']
                attention_mask = encoded.get('attention_mask', torch.ones_like(input_ids))
                
                # token_type_ids 처리 수정
                if 'token_type_ids' not in encoded or encoded['token_type_ids'] is None:
                    # 모델의 type_vocab_size 확인
                    type_vocab_size = getattr(self.model.config, 'type_vocab_size', 2)
                    token_type_ids = torch.zeros_like(input_ids)
                    # 모든 토큰을 첫 번째 타입(0)으로 설정
                    token_type_ids = token_type_ids.clamp(0, type_vocab_size - 1)
                else:
                    token_type_ids = encoded['token_type_ids']
                    # token_type_ids 값이 type_vocab_size를 초과하지 않도록 제한
                    type_vocab_size = getattr(self.model.config, 'type_vocab_size', 2)
                    token_type_ids = token_type_ids.clamp(0, type_vocab_size - 1)
                    
                # 입력을 디바이스로 이동
                inputs = {
                    'input_ids': input_ids.to(self.device),
                    'attention_mask': attention_mask.to(self.device),
                    'token_type_ids': token_type_ids.to(self.device)
                }
                
                # 입력 크기 검증
                if input_ids.size(1) > 512:
                    logger.warning(f"입력 크기가 너무 큽니다: {input_ids.size(1)} 토큰")
                    return 1, 0.0
                    
            except Exception as e:
                logger.error(f"토큰화 중 오류 발생: {str(e)}")
                logger.error(traceback.format_exc())
                return 1, 0.0
            
            # 모델 추론
            try:
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    pred = torch.argmax(probs, dim=-1).item()
                    confidence = probs[0][pred].item()
                    
                # 신뢰도 검증
                if confidence < 0.1:  # 신뢰도가 너무 낮은 경우
                    logger.warning(f"신뢰도가 너무 낮습니다: {confidence:.3f}")
                    return 1, 0.0
                    
                return pred, confidence
                
            except Exception as e:
                logger.error(f"모델 추론 중 오류 발생: {str(e)}")
                logger.error(traceback.format_exc())
                return 1, 0.0
            
        except Exception as e:
            logger.error(f"감성 분석 중 예상치 못한 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            return 1, 0.0  # 오류 발생 시 중립으로 처리

    def analyze_text(self, text: str) -> Tuple[int, float]:
        """predict 메서드의 별칭"""
        return self.predict(text)
    
    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리 및 길이 제한"""
        if not text or not isinstance(text, str):
            return ""
        
        try:
            # HTML 태그 제거
            text = re.sub(r'<[^>]+>', '', text)
            # URL 제거
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
            # 이메일 제거
            text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', ' ', text)
            # 특수문자 처리 (한글, 영문, 숫자, 기본 문장부호만 유지)
            text = re.sub(r'[^\w\s가-힣.,!?]', ' ', text)
            # 연속된 공백 제거
            text = re.sub(r'\s+', ' ', text)
            # 앞뒤 공백 제거
            text = text.strip()
            
            # 문자 수 제한
            if len(text) > 2048:
                text = text[:2048]
                logger.warning("텍스트가 2048자를 초과하여 잘립니다.")
            
            return text
            
        except Exception as e:
            logger.error(f"텍스트 전처리 중 오류 발생: {str(e)}")
            return ""
