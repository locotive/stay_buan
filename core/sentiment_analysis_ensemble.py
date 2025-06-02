from core.sentiment_analysis_kobert import KoBERTSentimentAnalyzer
from core.sentiment_analysis_kcelectra import KcELECTRASentimentAnalyzer
from core.sentiment_analysis_kosentencebert import KoSentenceBERTSentimentAnalyzer
from core.sentiment_analysis_kcbert_large import KcBERTLargeSentimentAnalyzer
from collections import Counter
from typing import List, Dict, Tuple, Union
import logging
import torch
import numpy as np
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 로깅 핸들러 설정
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class EnsembleSentimentAnalyzer:
    """다양한 감성 분석 모델 앙상블"""
    
    _instance = None
    
    def __new__(cls, models: List = None):
        if cls._instance is None:
            cls._instance = super(EnsembleSentimentAnalyzer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, models: List = None):
        # 모델 목록이 변경되었거나 초기화되지 않은 경우에만 재초기화
        if not hasattr(self, '_initialized') or not self._initialized or (models is not None and set(models) != set(self.model_names)):
            self.model_mapping = {
                'kobert': lambda: KoBERTSentimentAnalyzer(model_path="data/models/kobert"),
                'kcelectra-base-v2022': lambda: KcELECTRASentimentAnalyzer(model_path="data/models/kcelectra-base-v2022"),
                'kcelectra': lambda: KcELECTRASentimentAnalyzer(model_path="data/models/kcelectra"),
                'kcbert-large': lambda: KcBERTLargeSentimentAnalyzer(model_path="data/models/kcbert-large"),
                'kosentencebert': lambda: KoSentenceBERTSentimentAnalyzer(model_path="data/models/kosentencebert")
            }
            
            # 사용자가 지정한 모델 목록 사용 (없으면 모든 모델 사용)
            self.model_names = models if models else list(self.model_mapping.keys())
            
            # 지원하지 않는 모델 필터링
            self.model_names = [name for name in self.model_names if name in self.model_mapping]
            
            if not self.model_names:
                logger.warning("유효한 모델이 선택되지 않았습니다. 기본 모델을 사용합니다.")
                self.model_names = ['kobert', 'kcelectra-base-v2022', 'kcelectra']
            
            logger.info(f"선택된 모델 목록: {self.model_names}")
            self.analyzers = {}
            
            # 기존 모델 해제
            if hasattr(self, 'analyzers'):
                for analyzer in self.analyzers.values():
                    if hasattr(analyzer, 'model'):
                        del analyzer.model
                    if hasattr(analyzer, 'tokenizer'):
                        del analyzer.tokenizer
            
            # 모델 초기화
            self._initialize_models()
            
            # 초기화된 모델 수 확인
            initialized_models = list(self.analyzers.keys())
            if len(initialized_models) != len(self.model_names):
                failed_models = set(self.model_names) - set(initialized_models)
                logger.warning(f"일부 모델 초기화 실패: {failed_models}")
            
            logger.info(f"앙상블 분석기 초기화 완료 (모델 수: {len(self.analyzers)}, 초기화된 모델: {initialized_models})")
            
            # 레이블 매핑 정의 (수정)
            self.label_map = {
                0: "negative",  # 0을 부정으로 변경
                1: "neutral",
                2: "positive"   # 2를 긍정으로 변경
            }
            
            # 숫자 레이블 매핑 (수정)
            self.numeric_label_map = {
                "negative": 0,  # 부정을 0으로 변경
                "neutral": 1,
                "positive": 2   # 긍정을 2로 변경
            }
            
            self._initialized = True
    
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
                            'kcelectra-base-v2022': 'beomi/kcelectra-base-v2022',
                            'kcelectra': 'beomi/kcelectra-base',
                            'kcbert-large': 'beomi/kcbert-large',
                            'kosentencebert': 'snunlp/KR-SBERT-V40K-klueNLI-augSTS'
                        }[model_name]
                        
                        # 모델 다운로드 시도
                        logger.info(f"{model_name} 모델 다운로드 시작: {repo_id}")
                        snapshot_download(
                            repo_id=repo_id,
                            local_dir=model_path,
                            local_dir_use_symlinks=False,
                            ignore_patterns=["*.md", "*.txt", "*.json", "*.h5", "*.ckpt", "*.bin", "*.pt", "*.pth"]
                        )
                        logger.info(f"{model_name} 모델 다운로드 완료")
                        
                    except Exception as e:
                        logger.error(f"{model_name} 모델 다운로드 실패: {str(e)}")
                        continue
                
                # 모델 초기화
                try:
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
                
            except Exception as e:
                logger.error(f"{model_name} 모델 처리 중 예상치 못한 오류 발생: {str(e)}")
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
        
        logger.info(f"입력 텍스트: {text[:100]}...")  # 텍스트 일부 로깅
        
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
                
                # 원본 예측 결과 상세 로깅
                logger.info(f"{model_name} 모델 예측 결과:")
                logger.info(f"  - 레이블: {pred} ({self.label_map.get(pred, 'unknown')})")
                logger.info(f"  - 신뢰도: {conf:.3f}")
                
                predictions.append(pred)
                confidences.append(conf)
                
            except Exception as e:
                logger.error(f"{model_name} 모델 예측 중 오류 발생: {str(e)}")
                continue

        if not predictions:
            logger.warning("모든 모델 예측 실패")
            return 1, 0.0  # neutral, 0.0 confidence

        # 다수결 투표 결과 상세 로깅
        label_counts = {}
        for pred, conf in zip(predictions, confidences):
            if pred not in label_counts:
                label_counts[pred] = {'count': 0, 'conf_sum': 0.0}
            label_counts[pred]['count'] += 1
            label_counts[pred]['conf_sum'] += conf
            
        logger.info("각 레이블별 투표 결과:")
        for label, info in label_counts.items():
            avg_conf = info['conf_sum'] / info['count']
            logger.info(f"  - {self.label_map.get(label, 'unknown')}: {info['count']}표 (평균 신뢰도: {avg_conf:.3f})")

        # 가장 많은 표를 받은 라벨 선택
        max_count = max(label_counts.values(), key=lambda x: x['count'])['count']
        candidates = [label for label, info in label_counts.items() if info['count'] == max_count]
        
        # 동률인 경우 평균 신뢰도가 높은 라벨 선택
        if len(candidates) > 1:
            selected_label = max(candidates, 
                               key=lambda x: label_counts[x]['conf_sum'] / label_counts[x]['count'])
            logger.info(f"동률 발생: {[self.label_map.get(c, 'unknown') for c in candidates]}")
        else:
            selected_label = candidates[0]
            
        # 평균 신뢰도 계산
        avg_confidence = label_counts[selected_label]['conf_sum'] / label_counts[selected_label]['count']
        
        # 최종 예측 결과 로깅
        logger.info(f"앙상블 최종 예측 결과: {selected_label} ({self.label_map.get(selected_label, 'unknown')}, 신뢰도: {avg_confidence:.3f})")
        return selected_label, avg_confidence

    def analyze_text(self, text: str) -> Tuple[int, float]:
        """analyze_text는 predict의 별칭으로 사용"""
        return self.predict(text)
