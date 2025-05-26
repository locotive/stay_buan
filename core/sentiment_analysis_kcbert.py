import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
from typing import Optional, Tuple
import os
import re

logger = logging.getLogger(__name__)

class KCBERTSentimentAnalyzer:
    """KCBERT 기반 감성 분석"""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(KCBERTSentimentAnalyzer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, model_path: str = None):
        if hasattr(self, 'initialized'):
            return
            
        self.model_path = model_path or "data/models/kcbert"
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
            
            logger.info("KCBERT 모델 초기화 완료")
            
            self.initialized = True
            
        except Exception as e:
            logger.error(f"KCBERT 모델 초기화 중 오류 발생: {str(e)}")
            raise
    
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

    def predict(self, text: str) -> Tuple[int, float]:
        """감성 분석 예측 수행"""
        try:
            if not text or not isinstance(text, str):
                logger.warning("유효하지 않은 입력 텍스트")
                return 1, 0.0  # neutral, 0.0 confidence

            # 텍스트 전처리
            text = self.preprocess_text(text)
            if not text:
                logger.warning("전처리 후 텍스트가 비어있습니다")
                return 1, 0.0

            # 토큰화
            try:
                # 토큰화 및 입력 준비 (단계별로 수행)
                encoded = self.tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    add_special_tokens=True,
                    return_tensors="pt"
                )
                
                # 필수 입력 필드 확인 및 생성
                input_ids = encoded['input_ids']
                attention_mask = encoded.get('attention_mask', torch.ones_like(input_ids))
                
                # token_type_ids가 없는 경우 생성
                if 'token_type_ids' not in encoded:
                    token_type_ids = torch.zeros_like(input_ids)
                else:
                    token_type_ids = encoded['token_type_ids']
                    
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
                return 1, 0.0

            # 모델 예측
            try:
                with torch.no_grad():
                    # 모델 입력 크기 확인
                    if hasattr(self.model, 'config'):
                        max_length = getattr(self.model.config, 'max_position_embeddings', 512)
                        if input_ids.size(1) > max_length:
                            logger.warning(f"모델의 최대 입력 길이({max_length})를 초과합니다")
                            return 1, 0.0
                    
                    outputs = self.model(**inputs)
                    logits = outputs.logits.detach().cpu()
                    probs = torch.softmax(logits, dim=-1)
                    pred_label = torch.argmax(probs, dim=-1).item()
                    confidence = probs[0][pred_label].item()
                    
                    # 신뢰도 검증
                    if confidence < 0.1:  # 신뢰도가 너무 낮은 경우
                        logger.warning(f"신뢰도가 너무 낮습니다: {confidence:.3f}")
                        return 1, 0.0
                    
                    return pred_label, confidence
                
            except Exception as e:
                logger.error(f"모델 예측 중 오류 발생: {str(e)}")
                return 1, 0.0

        except Exception as e:
            logger.error(f"KCBERT 예측 중 예상치 못한 오류 발생: {str(e)}")
            return 1, 0.0

    def analyze_text(self, text: str) -> Tuple[int, float]:
        """analyze_text는 predict의 별칭으로 사용"""
        return self.predict(text)

class KcBERTLargeSentimentAnalyzer:
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            # 동일한 모델 ID 사용
            model_id = "beomi/kcbert-large"
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                max_length=512,
                truncation=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                trust_remote_code=True,
                num_labels=3
            ).to(self.device)
            self.model.eval()
            logger.info(f"KcBERT-large 모델 초기화 완료 (device: {self.device})")
        except Exception as e:
            logger.error(f"KcBERT-large 모델 초기화 중 오류 발생: {str(e)}")
            raise

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

    def predict(self, text: str) -> Tuple[int, float]:
        """감성 분석 예측 수행"""
        try:
            if not text or not isinstance(text, str):
                logger.warning("유효하지 않은 입력 텍스트")
                return 1, 0.0  # neutral, 0.0 confidence

            # 텍스트 전처리
            text = self.preprocess_text(text)
            if not text:
                logger.warning("전처리 후 텍스트가 비어있습니다")
                return 1, 0.0

            # 토큰화
            try:
                # 토큰화 및 입력 준비 (단계별로 수행)
                encoded = self.tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    add_special_tokens=True,
                    return_tensors="pt"
                )
                
                # 필수 입력 필드 확인 및 생성
                input_ids = encoded['input_ids']
                attention_mask = encoded.get('attention_mask', torch.ones_like(input_ids))
                
                # token_type_ids가 없는 경우 생성
                if 'token_type_ids' not in encoded:
                    token_type_ids = torch.zeros_like(input_ids)
                else:
                    token_type_ids = encoded['token_type_ids']
                    
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
                return 1, 0.0

            # 모델 예측
            try:
                with torch.no_grad():
                    # 모델 입력 크기 확인
                    if hasattr(self.model, 'config'):
                        max_length = getattr(self.model.config, 'max_position_embeddings', 512)
                        if input_ids.size(1) > max_length:
                            logger.warning(f"모델의 최대 입력 길이({max_length})를 초과합니다")
                            return 1, 0.0
                    
                    outputs = self.model(**inputs)
                    logits = outputs.logits.detach().cpu()
                    probs = torch.softmax(logits, dim=-1)
                    pred_label = torch.argmax(probs, dim=-1).item()
                    confidence = probs[0][pred_label].item()
                    
                    # 신뢰도 검증
                    if confidence < 0.1:  # 신뢰도가 너무 낮은 경우
                        logger.warning(f"신뢰도가 너무 낮습니다: {confidence:.3f}")
                        return 1, 0.0
                    
                    return pred_label, confidence
                
            except Exception as e:
                logger.error(f"모델 예측 중 오류 발생: {str(e)}")
                return 1, 0.0

        except Exception as e:
            logger.error(f"KcBERT-large 예측 중 예상치 못한 오류 발생: {str(e)}")
            return 1, 0.0

    def analyze_text(self, text: str) -> Tuple[int, float]:
        """analyze_text는 predict의 별칭으로 사용"""
        return self.predict(text) 