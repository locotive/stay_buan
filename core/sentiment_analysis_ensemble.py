from core.sentiment_analysis_kobert import KoBERTSentimentAnalyzer
from core.sentiment_analysis_kcbert import KCBERTSentimentAnalyzer, KcBERTLargeSentimentAnalyzer
from core.sentiment_analysis_kcelectra import KcELECTRASentimentAnalyzer
from core.sentiment_analysis_kosentencebert import KoSentenceBERTSentimentAnalyzer
from collections import Counter
from typing import List, Dict, Tuple, Union
import logging
import torch
import numpy as np
import os

logger = logging.getLogger(__name__)

class EnsembleSentimentAnalyzer:
    """다양한 감성 분석 모델 앙상블 - 싱글톤 패턴"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, models: List = None):
        if cls._instance is None:
            cls._instance = super(EnsembleSentimentAnalyzer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, models: List = None):
        if not EnsembleSentimentAnalyzer._initialized:
            self.model_mapping = {
                'kobert': lambda: KoBERTSentimentAnalyzer(model_path="data/models/kobert"),
                'kcbert': lambda: KCBERTSentimentAnalyzer(model_path="data/models/kcbert"),
                'kcelectra': lambda: KcELECTRASentimentAnalyzer(model_path="data/models/kcelectra"),
                'kcbert-large': lambda: KcBERTLargeSentimentAnalyzer(model_path="data/models/kcbert-large"),
                'kosentencebert': lambda: KoSentenceBERTSentimentAnalyzer(model_path="data/models/kosentencebert")
            }
            
            # 기본 모델 조합
            self.model_names = ['kobert', 'kcbert', 'kcelectra']
            self.analyzers = {}
            
            # 모델 초기화
            self._initialize_models()
            logger.info(f"앙상블 분석기 초기화 완료 (모델 수: {len(self.analyzers)})")
            
            # 레이블 매핑 정의
            self.label_map = {
                0: "negative",
                1: "neutral",
                2: "positive"
            }
            
            # 숫자 레이블 매핑
            self.numeric_label_map = {
                "negative": 0,
                "neutral": 1,
                "positive": 2
            }
            
            EnsembleSentimentAnalyzer._initialized = True
    
    def _initialize_models(self):
        """모델 초기화 및 메서드 검증"""
        for model_name in self.model_names:
            try:
                if model_name not in self.model_mapping:
                    logger.warning(f"지원하지 않는 모델: {model_name}")
                    continue
                
                # 모델 디렉토리 확인
                model_path = f"data/models/{model_name}"
                if not os.path.exists(model_path):
                    logger.warning(f"{model_name} 모델 파일이 없습니다. 다운로드를 시도합니다.")
                    try:
                        from huggingface_hub import snapshot_download
                        repo_id = {
                            'kobert': 'monologg/kobert',
                            'kcbert': 'beomi/kcbert-base',
                            'kcelectra': 'beomi/kcelectra-base',
                            'kcbert-large': 'beomi/kcbert-large',
                            'kosentencebert': 'snunlp/KR-SBERT-V40K-klueNLI-augSTS'
                        }[model_name]
                        snapshot_download(
                            repo_id=repo_id,
                            local_dir=model_path,
                            local_dir_use_symlinks=False
                        )
                    except Exception as e:
                        logger.error(f"{model_name} 모델 다운로드 실패: {str(e)}")
                        continue
                
                # 모델 초기화
                analyzer = self.model_mapping[model_name]()
                
                # predict 또는 analyze_text 메서드 존재 여부 확인
                if not (hasattr(analyzer, 'predict') or hasattr(analyzer, 'analyze_text')):
                    logger.warning(f"{model_name} 모델에 predict 또는 analyze_text 메서드가 없습니다")
                    continue
                
                self.analyzers[model_name] = analyzer
                logger.info(f"{model_name} 모델 초기화 완료")
                
            except Exception as e:
                logger.error(f"{model_name} 모델 초기화 중 오류 발생: {str(e)}")
                continue
    
    def predict(self, text: str) -> Tuple[int, float]:
        """앙상블 기반 감성 분석 예측"""
        if not text or not isinstance(text, str):
            logger.warning("유효하지 않은 입력 텍스트")
            return 1, 0.0  # neutral, 0.0 confidence

        if not self.analyzers:
            logger.warning("초기화된 모델이 없습니다")
            return 1, 0.0  # neutral, 0.0 confidence

        predictions = []
        confidences = []
        
        for model_name, analyzer in self.analyzers.items():
            try:
                # predict 또는 analyze_text 메서드 사용
                if hasattr(analyzer, 'predict'):
                    pred, conf = analyzer.predict(text)
                elif hasattr(analyzer, 'analyze_text'):
                    pred, conf = analyzer.analyze_text(text)
                else:
                    logger.warning(f"{model_name} 모델에 예측 메서드가 없습니다")
                    continue
                
                predictions.append(pred)
                confidences.append(conf)
                logger.debug(f"{model_name} 예측 결과: {pred} (신뢰도: {conf:.3f})")
                
            except Exception as e:
                logger.error(f"{model_name} 모델 예측 중 오류 발생: {str(e)}")
                continue

        if not predictions:
            logger.warning("모든 모델 예측 실패")
            return 1, 0.0  # neutral, 0.0 confidence

        # 다수결 투표
        label_counts = {}
        for pred, conf in zip(predictions, confidences):
            if pred not in label_counts:
                label_counts[pred] = {'count': 0, 'conf_sum': 0.0}
            label_counts[pred]['count'] += 1
            label_counts[pred]['conf_sum'] += conf

        # 가장 많은 표를 받은 라벨 선택
        max_count = max(label_counts.values(), key=lambda x: x['count'])['count']
        candidates = [label for label, info in label_counts.items() if info['count'] == max_count]
        
        # 동률인 경우 평균 신뢰도가 높은 라벨 선택
        if len(candidates) > 1:
            selected_label = max(candidates, 
                               key=lambda x: label_counts[x]['conf_sum'] / label_counts[x]['count'])
        else:
            selected_label = candidates[0]
            
        # 평균 신뢰도 계산
        avg_confidence = label_counts[selected_label]['conf_sum'] / label_counts[selected_label]['count']
        
        logger.info(f"앙상블 예측 결과: {selected_label} (신뢰도: {avg_confidence:.3f})")
        return selected_label, avg_confidence

    def analyze_text(self, text: str) -> Tuple[int, float]:
        """analyze_text는 predict의 별칭으로 사용"""
        return self.predict(text)
