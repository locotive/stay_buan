import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from functools import lru_cache
import json
from datetime import datetime
import os
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DataProcessor:
    """데이터 전처리 파이프라인 클래스"""
    
    def __init__(self):
        self.sentiment_map = {
            0: 'negative',
            1: 'neutral',
            2: 'positive',
            'negative': 'negative',
            'neutral': 'neutral',
            'positive': 'positive'
        }
        self.reverse_sentiment_map = {v: k for k, v in self.sentiment_map.items()}
        self.cache_dir = "data/cache"
        self.model_dir = "data/models"  # 모델 파일 저장 디렉토리
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 모델 설정 (KoAlpaca 제외, 새로운 모델 추가)
        self.model_configs = {
            'kobert': {
                'name': 'monologg/kobert',
                'local_path': os.path.join(self.model_dir, 'kobert'),
                'offline': True
            },
            'kcbert': {
                'name': 'beomi/kcbert-base',
                'local_path': os.path.join(self.model_dir, 'kcbert'),
                'offline': True
            },
            'kcelectra': {
                'name': 'beomi/kcelectra-base',
                'local_path': os.path.join(self.model_dir, 'kcelectra'),
                'offline': True
            },
            'kcbert-large': {
                'name': 'beomi/kcbert-large',
                'local_path': os.path.join(self.model_dir, 'kcbert-large'),
                'offline': True
            },
            'kosentencebert': {
                'name': 'snunlp/KR-SBERT-V40K-klueNLI-augSTS',
                'local_path': os.path.join(self.model_dir, 'kosentencebert'),
                'offline': True
            }
        }
        
        # 감성 분석 필터링 조건 수정
        self.sentiment_filters = {
            'min_text_length': 30,
            'max_text_length': 20000,
            'min_confidence': 0.2,  # 신뢰도 임계값을 0.2로 낮춤
            'exclude_keywords': ['광고', '홍보', 'sponsored'],
            'required_keywords': [],
            'min_content_words': 3
        }
        
        # 사용 가능한 감성분석 모델 목록
        self.available_models = {
            'kobert': 'KoBERT (가벼운 모델)',
            'kcbert': 'KCBERT (중간 크기 모델)',
            'kcelectra': 'KcELECTRA (감성분석 특화)',
            'kcbert-large': 'KcBERT-large (큰 모델)',
            'kosentencebert': 'KoSentenceBERT (문장 수준 분석)'
        }
        
        # 미리 정의된 모델 조합
        self.model_combinations = {
            'light': ['kobert', 'kcelectra'],  # 가벼운 조합
            'balanced': ['kcbert', 'kcelectra', 'kosentencebert'],  # 균형잡힌 조합
            'heavy': ['kcbert-large', 'kosentencebert', 'kcelectra'],  # 정확도 중심
            'custom': []  # 사용자 정의
        }
        
        # 모델 초기화 상태
        self._initialized_models = {}
    
    def _download_model(self, model_name: str) -> bool:
        """모델 파일 다운로드"""
        try:
            from huggingface_hub import snapshot_download
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            config = self.model_configs[model_name]
            if os.path.exists(config['local_path']):
                logger.info(f"{model_name} 모델이 이미 로컬에 존재합니다.")
                return True
                
            logger.info(f"{model_name} 모델 다운로드 시작...")
            
            # 모델 파일 다운로드
            snapshot_download(
                repo_id=config['name'],
                local_dir=config['local_path'],
                local_dir_use_symlinks=False
            )
            
            # 토크나이저와 모델 로드 테스트
            tokenizer = AutoTokenizer.from_pretrained(config['local_path'])
            model = AutoModelForSequenceClassification.from_pretrained(
                config['local_path']
            )
            
            logger.info(f"{model_name} 모델 다운로드 완료")
            return True
            
        except Exception as e:
            logger.error(f"{model_name} 모델 다운로드 실패: {str(e)}")
            return False
    
    def _initialize_model(self, model_name: str):
        """모델 초기화"""
        if model_name in self._initialized_models:
            return self._initialized_models[model_name]
            
        try:
            config = self.model_configs[model_name]
            
            # 모델 파일 확인 및 다운로드
            if not os.path.exists(config['local_path']):
                if not self._download_model(model_name):
                    raise RuntimeError(f"{model_name} 모델 초기화 실패")
            
            # 모델 클래스 동적 임포트
            if model_name == 'kobert':
                from core.sentiment_analysis_kobert import KoBERTSentimentAnalyzer
                model = KoBERTSentimentAnalyzer(model_path=config['local_path'])
            elif model_name == 'kcbert':
                from core.sentiment_analysis_kcbert import KCBERTSentimentAnalyzer
                model = KCBERTSentimentAnalyzer(model_path=config['local_path'])
            elif model_name == 'kcelectra':
                from core.sentiment_analysis_kcelectra import KcELECTRASentimentAnalyzer
                model = KcELECTRASentimentAnalyzer(model_path=config['local_path'])
            elif model_name == 'kcbert-large':
                from core.sentiment_analysis_kcbert_large import KcBERTLargeSentimentAnalyzer
                model = KcBERTLargeSentimentAnalyzer(model_path=config['local_path'])
            elif model_name == 'kosentencebert':
                from core.sentiment_analysis_kosentencebert import KoSentenceBERTSentimentAnalyzer
                model = KoSentenceBERTSentimentAnalyzer(model_path=config['local_path'])
            else:
                raise ValueError(f"지원하지 않는 모델: {model_name}")
            
            self._initialized_models[model_name] = model
            logger.info(f"{model_name} 모델 초기화 완료")
            return model
            
        except Exception as e:
            logger.error(f"{model_name} 모델 초기화 중 오류 발생: {str(e)}")
            raise
    
    def get_available_models(self) -> Dict[str, str]:
        """사용 가능한 감성분석 모델 목록 반환"""
        return self.available_models
    
    def get_matching_processed_data(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """선택된 데이터셋과 매칭되는 processed 데이터 찾기"""
        try:
            processed_dir = "data/processed"
            if not os.path.exists(processed_dir):
                return None
                
            # 데이터셋 이름에서 플랫폼 추출 (예: naver_processed_20250514_025335.csv -> naver)
            platform = dataset_name.split('_')[0]
            
            # 해당 플랫폼의 가장 최근 processed 파일 찾기
            matching_files = [f for f in os.listdir(processed_dir) 
                            if f.startswith(f"{platform}_processed_") and f.endswith('.csv')]
            
            if not matching_files:
                return None
                
            # 가장 최근 파일 선택
            latest_file = sorted(matching_files)[-1]
            file_path = os.path.join(processed_dir, latest_file)
            
            # 데이터 로드
            df = pd.read_csv(file_path)
            logger.info(f"매칭되는 processed 데이터를 찾았습니다: {latest_file}")
            return df
            
        except Exception as e:
            logger.error(f"processed 데이터 검색 중 오류 발생: {str(e)}")
            return None
    
    def get_sentiment_analyzer(self, model_names: List[str] = None):
        """선택된 모델들로 앙상블 감성분석기 반환"""
        try:
            from core.sentiment_analysis_ensemble import EnsembleSentimentAnalyzer
            
            # 기본값 설정
            if model_names is None:
                model_names = ['kobert', 'kcbert']
            
            # 선택된 모델들 초기화
            models = [self._initialize_model(name) for name in model_names]
            
            return EnsembleSentimentAnalyzer(models=models)
                
        except Exception as e:
            logger.error(f"감성분석기 초기화 중 오류 발생: {str(e)}")
            raise
    
    def analyze_dataset(self, dataset_name: str, model_name: str) -> pd.DataFrame:
        """데이터셋 분석 실행"""
        try:
            logger.info(f"데이터셋 분석 시작: {dataset_name}, 모델: {model_name}")
            
            # 1. 매칭되는 processed 데이터 확인
            df = self.get_matching_processed_data(dataset_name)
            
            if df is not None:
                logger.info(f"기존 processed 데이터를 찾았습니다. 크기: {len(df)}")
                # 이미 감성분석이 되어있는지 확인
                if 'sentiment_label' in df.columns:
                    logger.info("이미 감성분석이 완료된 데이터입니다.")
                    return df
                else:
                    logger.info("감성분석을 시작합니다...")
                    analyzer = self.get_sentiment_analyzer(model_name)
                    return self.process_data(df, dataset_name.split('_')[0], analyzer)
            
            # 2. 매칭되는 데이터가 없는 경우
            logger.info("새로운 데이터 처리를 시작합니다...")
            raw_data_path = os.path.join("data/raw", dataset_name)
            if not os.path.exists(raw_data_path):
                logger.error(f"데이터셋을 찾을 수 없습니다: {raw_data_path}")
                raise FileNotFoundError(f"데이터셋을 찾을 수 없습니다: {dataset_name}")
            
            # 원본 데이터 로드
            logger.info(f"원본 데이터 로드 시도: {raw_data_path}")
            try:
                df = pd.read_csv(raw_data_path)
                logger.info(f"원본 데이터 로드 성공. 크기: {len(df)}")
            except Exception as e:
                logger.error(f"CSV 로드 실패, JSON 시도: {str(e)}")
                try:
                    with open(raw_data_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    df = pd.DataFrame(data)
                    logger.info(f"JSON 데이터 로드 성공. 크기: {len(df)}")
                except Exception as e:
                    logger.error(f"JSON 로드도 실패: {str(e)}")
                    raise
            
            if df.empty:
                logger.error("로드된 데이터가 비어있습니다.")
                return df
            
            # 필수 컬럼 확인
            required_columns = ['title', 'content', 'published_date']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"필수 컬럼 누락: {missing_columns}")
                raise ValueError(f"필수 컬럼이 누락되었습니다: {missing_columns}")
            
            # 감성분석기 초기화
            logger.info(f"감성분석기 초기화: {model_name}")
            analyzer = self.get_sentiment_analyzer(model_name)
            
            # 데이터 처리 및 감성분석
            logger.info("데이터 처리 및 감성분석 시작")
            processed_df = self.process_data(df, dataset_name.split('_')[0], analyzer)
            
            if processed_df.empty:
                logger.error("처리 후 데이터가 비어있습니다.")
            else:
                logger.info(f"데이터 처리 완료. 최종 크기: {len(processed_df)}")
            
            return processed_df
            
        except Exception as e:
            logger.error(f"데이터셋 분석 중 오류 발생: {str(e)}", exc_info=True)
            raise
    
    def is_valid_for_sentiment(self, text: str) -> Tuple[bool, str]:
        """감성 분석 대상 텍스트 검증"""
        if not isinstance(text, str):
            return False, "텍스트가 아닌 데이터"
            
        text = text.strip()
        
        # 길이 검증
        if len(text) < self.sentiment_filters['min_text_length']:
            return False, f"텍스트가 너무 짧음 ({len(text)}자)"
        if len(text) > self.sentiment_filters['max_text_length']:
            return False, f"텍스트가 너무 김 ({len(text)}자)"
            
        # 제외 키워드 검사
        for keyword in self.sentiment_filters['exclude_keywords']:
            if keyword in text:
                return False, f"제외 키워드 포함: {keyword}"
                
        # 필수 키워드 검사
        for keyword in self.sentiment_filters['required_keywords']:
            if keyword not in text:
                return False, f"필수 키워드 누락: {keyword}"
                
        return True, "유효한 텍스트"
    
    @lru_cache(maxsize=1000)  # 캐시 크기 증가
    def get_cached_sentiment(self, text: str, analyzer_name: str) -> Optional[tuple]:
        """감성 분석 결과 캐싱"""
        try:
            cache_file = os.path.join(self.cache_dir, f"sentiment_cache_{analyzer_name}.json")
            
            # 캐시 파일 로드
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
            else:
                cache = {}
            
            # 캐시된 결과가 있으면 반환
            if text in cache:
                return tuple(cache[text])
            
            return None
            
        except Exception as e:
            logger.error(f"캐시 로드 중 오류: {str(e)}")
            return None
    
    def save_sentiment_cache(self, text: str, result: tuple, analyzer_name: str):
        """감성 분석 결과 캐시 저장"""
        try:
            cache_file = os.path.join(self.cache_dir, f"sentiment_cache_{analyzer_name}.json")
            
            # 캐시 파일 로드
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
            else:
                cache = {}
            
            # 결과 저장
            cache[text] = list(result)
            
            # 캐시 파일 저장
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"캐시 저장 중 오류: {str(e)}")
    
    def analyze_sentiment_batch(self, texts: List[str], analyzer) -> List[Tuple[int, float]]:
        """배치 단위 감성 분석"""
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for text in texts:
                # 텍스트 검증
                is_valid, reason = self.is_valid_for_sentiment(text)
                if not is_valid:
                    logger.warning(f"감성 분석 제외: {reason}")
                    results.append((1, 0.0))  # 중립으로 처리
                    continue
                    
                # 캐시 확인
                cached_result = self.get_cached_sentiment(text, analyzer.__class__.__name__)
                if cached_result:
                    results.append(cached_result)
                    continue
                    
                # 새로운 분석 요청
                futures.append(executor.submit(analyzer.predict, text))
            
            # 결과 수집
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"감성 분석 중 오류: {str(e)}")
                    results.append((1, 0.0))  # 오류 시 중립으로 처리
                    
        return results
    
    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        if not isinstance(text, str):
            return ""
            
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        
        # URL을 공백으로 대체 (제거 대신)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        
        # 특수문자 제거 (한글, 영문, 숫자만 남김)
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 앞뒤 공백 제거
        text = text.strip()
        
        return text

    def validate_content(self, content: str) -> bool:
        """컨텐츠 유효성 검사 (더 유연한 조건 적용)"""
        if not isinstance(content, str):
            return False
            
        # 전처리된 텍스트
        processed_text = self.preprocess_text(content)
        
        # 길이 검사 (최소 길이만 체크)
        if len(processed_text) < self.sentiment_filters['min_text_length']:
            return False
            
        # 단어 수 검사 (최소 단어 수만 체크)
        words = processed_text.split()
        if len(words) < self.sentiment_filters['min_content_words']:
            return False
            
        # 제외 키워드 검사 (URL 관련 키워드 제외)
        for keyword in self.sentiment_filters['exclude_keywords']:
            if keyword.lower() in processed_text.lower():
                return False
                
        return True

    def validate_platform_data(self, df: pd.DataFrame, platform: str) -> Dict:
        """플랫폼별 데이터 검증"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # 필수 필드 확인
        required_fields = {
            'naver': ['title', 'content', 'published_date', 'url'],
            'youtube': ['title', 'content', 'published_date', 'url'],
            'dcinside': ['title', 'content', 'published_date', 'url'],
            'fmkorea': ['title', 'content', 'published_date', 'url'],
            'buan': ['title', 'content', 'published_date', 'url']
        }
        
        for field in required_fields.get(platform, []):
            if field not in df.columns:
                validation_results['is_valid'] = False
                validation_results['errors'].append(f"필수 필드 누락: {field}")
        
        # 데이터 타입 검증
        if 'published_date' in df.columns:
            invalid_dates = df[~df['published_date'].astype(str).str.match(r'^\d{8}$')]
            if not invalid_dates.empty:
                validation_results['warnings'].append(f"잘못된 날짜 형식: {len(invalid_dates)}개")
        
        # 컨텐츠 길이 검증
        if 'content' in df.columns:
            short_content = df[df['content'].str.len() < 10]
            if not short_content.empty:
                validation_results['warnings'].append(f"짧은 컨텐츠: {len(short_content)}개")
        
        return validation_results
    
    def process_data(self, df: pd.DataFrame, platform: str, analyzer=None) -> pd.DataFrame:
        """데이터 전처리 파이프라인 실행"""
        if df is None or df.empty:
            logger.warning("입력 데이터가 비어있습니다.")
            return pd.DataFrame()
            
        try:
            # 1. 데이터 검증
            validation_results = self.validate_platform_data(df, platform)
            
            # 검증 결과 로깅
            if not validation_results['is_valid']:
                logger.warning(f"데이터 검증 실패: {validation_results['errors']}")
            if validation_results['warnings']:
                logger.warning(f"데이터 검증 경고: {validation_results['warnings']}")
            
            # 2. 텍스트 전처리
            df['content'] = df['content'].apply(self.preprocess_text)
            df['title'] = df['title'].apply(self.preprocess_text)
            
            # 3. 감성 분석 (필수 수행)
            if df.empty:
                logger.warning("전처리 후 데이터가 비어있습니다.")
                return df
                
            # 감성 분석기 초기화 (없는 경우)
            if analyzer is None:
                from core.sentiment_analysis_ensemble import EnsembleSentimentAnalyzer
                analyzer = EnsembleSentimentAnalyzer()
                logger.info("EnsembleSentimentAnalyzer를 초기화했습니다.")
            
            # 배치 단위 감성 분석
            logger.info("감성 분석을 시작합니다...")
            sentiment_results = self.analyze_sentiment_batch(df['content'].tolist(), analyzer)
            
            # 결과 추가
            sentiment_df = pd.DataFrame(sentiment_results, columns=['sentiment', 'confidence'])
            df = pd.concat([df, sentiment_df], axis=1)
            
            # 감성 레이블 추가 (숫자와 문자열 모두 처리)
            df['sentiment_label'] = df['sentiment'].apply(lambda x: self.sentiment_map.get(x, 'neutral'))
            
            # 시각화를 위한 추가 컬럼
            df['sentiment_numeric'] = df['sentiment_label'].map(self.reverse_sentiment_map)
            df['sentiment_color'] = df['sentiment_label'].map({
                'positive': 'green',
                'neutral': 'gray',
                'negative': 'red'
            })
            
            # 신뢰도가 낮은 결과 필터링 (임계값 완화)
            df = df[df['confidence'] >= 0.3]  # 신뢰도 임계값을 0.3으로 낮춤
            
            # 감성 분석 후 데이터가 비어있는지 확인
            if df.empty:
                logger.warning("감성 분석 후 데이터가 비어있습니다. 신뢰도 임계값을 확인해주세요.")
                return df
                
            logger.info(f"감성 분석 완료: {len(df)}개 항목 처리됨")
            
            # 4. 날짜 정규화
            if 'published_date' in df.columns:
                df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
                # 유효하지 않은 날짜 제거
                df = df.dropna(subset=['published_date'])
                
                # 날짜 필터링 후 데이터가 비어있는지 확인
                if df.empty:
                    logger.warning("날짜 정규화 후 데이터가 비어있습니다. 날짜 형식을 확인해주세요.")
                    return df
            
            # 5. 중복 제거
            original_len = len(df)
            df = df.drop_duplicates(subset=['title', 'content', 'published_date'])
            if len(df) < original_len:
                logger.info(f"중복 제거: {original_len - len(df)}개 항목 제거됨")
            
            return df
            
        except Exception as e:
            logger.error(f"데이터 처리 중 오류 발생: {str(e)}", exc_info=True)
            return pd.DataFrame()
    
    def save_processed_data(self, df: pd.DataFrame, platform: str):
        """처리된 데이터 저장"""
        try:
            # 저장 디렉토리 생성
            save_dir = "data/processed"
            os.makedirs(save_dir, exist_ok=True)
            
            # 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{platform}_processed_{timestamp}.csv"
            filepath = os.path.join(save_dir, filename)
            
            # CSV로 저장
            df.to_csv(filepath, index=False, encoding='utf-8')
            
            # 메타데이터 저장
            metadata = {
                'platform': platform,
                'timestamp': timestamp,
                'row_count': len(df),
                'columns': list(df.columns),
                'sentiment_distribution': df['sentiment_label'].value_counts().to_dict() if 'sentiment_label' in df.columns else None,
                'filtering_stats': {
                    'min_text_length': self.sentiment_filters['min_text_length'],
                    'max_text_length': self.sentiment_filters['max_text_length'],
                    'min_confidence': self.sentiment_filters['min_confidence'],
                    'excluded_keywords': self.sentiment_filters['exclude_keywords'],
                    'required_keywords': self.sentiment_filters['required_keywords']
                }
            }
            
            metadata_path = os.path.join(save_dir, f"{platform}_metadata_{timestamp}.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            return filepath
            
        except Exception as e:
            logger.error(f"데이터 저장 중 오류 발생: {str(e)}")
            raise 