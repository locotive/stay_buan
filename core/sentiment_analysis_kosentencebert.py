from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class KoSentenceBERTSentimentAnalyzer:
    """KoSentenceBERT 기반 감성 분석 모델"""
    
    def __init__(self, model_path: str = None):
        """
        KoSentenceBERT 모델 초기화
        
        Args:
            model_path (str, optional): 로컬 모델 경로. None인 경우 Hugging Face에서 다운로드
        """
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path or "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
                local_files_only=model_path is not None
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path or "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
                num_labels=3,
                local_files_only=model_path is not None
            ).to(self.device)
            self.model.eval()
            logger.info("KoSentenceBERT 모델 초기화 완료")
        except Exception as e:
            logger.error(f"KoSentenceBERT 모델 초기화 중 오류 발생: {str(e)}")
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
            # 토큰화 및 모델 입력 준비
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length"
            ).to(self.device)
            
            # 추론
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                pred_label = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][pred_label].item()
            
            return pred_label, confidence
            
        except Exception as e:
            logger.error(f"KoSentenceBERT 예측 중 오류 발생: {str(e)}")
            return 1, 0.0  # 오류 시 중립으로 처리 