import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
from typing import Tuple, Optional
import os

logger = logging.getLogger(__name__)

class KoBERTSentimentAnalyzer:
    """KoBERT 기반 감성 분석 클래스"""
    
    def __init__(self, model_path="data/models/kobert"):
        """KoBERT 모델 초기화"""
        try:
            # 모델 파일이 없으면 자동 다운로드 시도
            if not (os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "pytorch_model.bin"))):
                from huggingface_hub import snapshot_download
                logger.info("모델 파일이 없어 자동 다운로드를 시도합니다.")
                snapshot_download(repo_id="skt/kobert-base-v1", local_dir=model_path, local_dir_use_symlinks=False)
            
            # 토크나이저와 모델 초기화
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3, trust_remote_code=True)
            
            # GPU 설정 및 모델 준비
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            self.model.eval()
                            
            logger.info(f"KoBERT 모델 초기화 완료 (device: {self.device})")
            
        except Exception as e:
            logger.error(f"KoBERT 모델 초기화 실패: {str(e)}", exc_info=True)
            raise
    
    def predict(self, text: str) -> Tuple[int, float]:
        """텍스트의 감성 분석 수행"""
        try:
            # 입력 텍스트 토크나이징
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
                pred = torch.argmax(probs, dim=1).item()
                conf = probs[0][pred].item()
                
                return pred, conf
                
        except Exception as e:
            logger.error(f"감성 분석 중 오류 발생: {str(e)}", exc_info=True)
            return 1, 0.0  # 오류 발생시 중립(1) 반환
