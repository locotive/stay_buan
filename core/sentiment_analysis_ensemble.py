from core.sentiment_analysis_kobert import KoBERTSentimentAnalyzer
from core.sentiment_analysis_kcbert import KCBERTSentimentAnalyzer
from core.sentiment_analysis_kcelectra import KcELECTRASentimentAnalyzer
from core.sentiment_analysis_kcbert_large import KcBERTLargeSentimentAnalyzer
from core.sentiment_analysis_kosentencebert import KoSentenceBERTSentimentAnalyzer
from collections import Counter
from typing import List, Dict, Tuple

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
            # 선택된 모델들만 사용
            self.analyzers = models if models is not None else [
                KoBERTSentimentAnalyzer(),
                KCBERTSentimentAnalyzer()
            ]
            
            # 레이블 매핑 정의
            self.label_map = {
                0: "negative",
                1: "neutral",
                2: "positive"
            }
            
            EnsembleSentimentAnalyzer._initialized = True
    
    def predict(self, text: str) -> Tuple[str, float, Dict[str, str]]:
        """
        앙상블 다수결 투표 기반 감성 분석 예측 수행
        
        Args:
            text (str): 분석할 텍스트
            
        Returns:
            tuple: (감성 레이블 문자열, 신뢰도 점수, 모델별 투표 결과 딕셔너리)
        """
        # 각 모델의 예측 결과 수집
        results = []
        model_votes = {}
        
        for analyzer in self.analyzers:
            try:
                label, conf = analyzer.predict(text)
                results.append((label, conf))
                model_votes[analyzer.__class__.__name__] = self.label_map[label]
            except Exception as e:
                logger.error(f"{analyzer.__class__.__name__} 예측 중 오류: {str(e)}")
                results.append((1, 0.0))  # 오류 시 중립으로 처리
                model_votes[analyzer.__class__.__name__] = "neutral"
        
        # 모든 모델이 실패한 경우 중립 반환
        if all(conf == 0.0 for _, conf in results):
            return "neutral", 0.0, model_votes
        
        # 다수결 투표로 최종 레이블 결정
        votes = [label for label, _ in results]
        vote_counter = Counter(votes)
        majority_label = vote_counter.most_common(1)[0][0]
        
        # 동점일 경우 신뢰도가 가장 높은 모델의 예측을 선택
        if len(vote_counter) == len(self.analyzers) and vote_counter.most_common(1)[0][1] == 1:
            confidences = {label: conf for label, conf in results}
            majority_label = max(confidences, key=confidences.get)
        
        # 최종 신뢰도 계산 (동일한 예측을 한 모델 수 / 전체 모델 수)
        confidence = vote_counter[majority_label] / len(self.analyzers)
        
        return self.label_map[majority_label], confidence, model_votes
