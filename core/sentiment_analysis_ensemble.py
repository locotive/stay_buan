from core.sentiment_analysis_kobert import KoBERTSentimentAnalyzer
from core.sentiment_analysis_koalpaca import KoAlpacaSentimentAnalyzer
from core.sentiment_analysis_kcbert import KCBERTSentimentAnalyzer
from collections import Counter

class EnsembleSentimentAnalyzer:
    """다양한 감성 분석 모델 앙상블 - 싱글톤 패턴"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EnsembleSentimentAnalyzer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not EnsembleSentimentAnalyzer._initialized:
            # 개별 모델 초기화 (오프라인/온라인 자동 다운로드 지원)
            self.kobert_analyzer = KoBERTSentimentAnalyzer()
            self.kcbert_analyzer = KCBERTSentimentAnalyzer()
            self.koalpaca_analyzer = KoAlpacaSentimentAnalyzer()
            
            # 레이블 매핑 정의
            self.label_map = {
                0: "negative",
                1: "neutral",
                2: "positive"
            }
            
            EnsembleSentimentAnalyzer._initialized = True
    
    def predict(self, text):
        """
        앙상블 다수결 투표 기반 감성 분석 예측 수행
        
        Args:
            text (str): 분석할 텍스트
            
        Returns:
            tuple: (감성 레이블 문자열, 신뢰도 점수, 모델별 투표 결과 딕셔너리)
        """
        # 각 모델의 예측 결과 수집
        kobert_result = self.kobert_analyzer.predict(text)
        kcbert_result = self.kcbert_analyzer.predict(text)
        koalpaca_result = self.koalpaca_analyzer.predict(text)
        
        # 각 모델의 예측 결과와 신뢰도 추출
        kobert_label, kobert_conf = kobert_result
        kcbert_label, kcbert_conf = kcbert_result
        koalpaca_label, koalpaca_conf = koalpaca_result
        
        # 모델별 투표 결과 저장
        model_votes = {
            "kobert": self.label_map[kobert_label],
            "kcbert": self.label_map[kcbert_label],
            "koalpaca": self.label_map[koalpaca_label]
        }
        
        # 모든 모델이 실패한 경우 중립 반환
        if kobert_conf == 0 and kcbert_conf == 0 and koalpaca_conf == 0:
            return "neutral", 0.0, model_votes
        
        # 다수결 투표로 최종 레이블 결정
        votes = [kobert_label, kcbert_label, koalpaca_label]
        vote_counter = Counter(votes)
        majority_label = vote_counter.most_common(1)[0][0]
        
        # 동점일 경우 신뢰도가 가장 높은 모델의 예측을 선택
        if len(vote_counter) == 3 and vote_counter.most_common(1)[0][1] == 1:
            confidences = {
                kobert_label: kobert_conf,
                kcbert_label: kcbert_conf,
                koalpaca_label: koalpaca_conf
            }
            majority_label = max(confidences, key=confidences.get)
        
        # 최종 신뢰도 계산 (동일한 예측을 한 모델 수 / 전체 모델 수)
        confidence = vote_counter[majority_label] / len(votes)
        
        # 문자열 레이블 반환
        sentiment_label = self.label_map[majority_label]
        
        return sentiment_label, confidence, model_votes
