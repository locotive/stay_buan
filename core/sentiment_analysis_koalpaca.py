from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class KoAlpacaSentimentAnalyzer:
    """KoAlpaca 모델을 사용한 감성 분석기 - 싱글톤 패턴"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KoAlpacaSentimentAnalyzer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not KoAlpacaSentimentAnalyzer._initialized:
            self.model_name = "beomi/KoAlpaca-Polyglot-12.8B"
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = None
            self.model = None
            KoAlpacaSentimentAnalyzer._initialized = True
    
    def _load_model(self):
        """지연 로딩 방식으로 모델 초기화"""
        if self.tokenizer is None or self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # 8비트 양자화를 사용하여 메모리 효율성 향상
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                load_in_8bit=True,
                device_map="auto"
            )
    
    def predict(self, text):
        """감성 분석 예측 수행"""
        self._load_model()
        
        # 프롬프트 구성
        prompt = f"""다음 문장의 감성을 분석해 주세요. 긍정적(positive), 중립적(neutral), 부정적(negative) 중 하나로 답변해주세요.

텍스트: {text}

감성:"""

        # 토큰화 및 추론
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 결과 디코딩 및 파싱
        result = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
        
        # 감성 레이블 매핑
        if "positive" in result:
            sentiment = 2  # 긍정
            confidence = 0.9  # 신뢰도는 고정값 사용
        elif "negative" in result:
            sentiment = 0  # 부정
            confidence = 0.9
        else:
            sentiment = 1  # 중립
            confidence = 0.7
        
        return sentiment, confidence 