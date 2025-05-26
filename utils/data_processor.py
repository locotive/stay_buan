import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from functools import lru_cache
import json
from datetime import datetime
import os
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pathlib import Path
from core.sentiment_analysis_ensemble import EnsembleSentimentAnalyzer

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
        
        self.analyzer = EnsembleSentimentAnalyzer()
    
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
    
    def analyze_dataset(self, 
                       input_file: Union[str, Path],
                       output_dir: Optional[Union[str, Path]] = None,
                       models: Optional[List[str]] = None,
                       confidence_threshold: float = 0.1) -> pd.DataFrame:
        """데이터셋 감성 분석 수행"""
        try:
            # 입력 파일 경로 처리
            if isinstance(input_file, str):
                # 상대 경로인 경우 data/raw 디렉토리 기준으로 처리
                if not os.path.isabs(input_file):
                    input_path = Path("data/raw") / input_file
                else:
                    input_path = Path(input_file)
            else:
                input_path = input_file

            if not input_path.exists():
                # 파일이 없는 경우 data/raw 디렉토리에서 다시 시도
                raw_path = Path("data/raw") / input_path.name
                if raw_path.exists():
                    input_path = raw_path
                else:
                    raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_file} (시도한 경로: {input_path}, {raw_path})")
                
            logger.info(f"데이터셋 분석 시작: {input_path.name} (경로: {input_path})")
            
            # 파일 형식에 따른 로드
            if input_path.suffix == '.json':
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
            elif input_path.suffix == '.csv':
                df = pd.read_csv(input_path, encoding='utf-8')
            else:
                raise ValueError(f"지원하지 않는 파일 형식: {input_path.suffix}")
            
            # 데이터셋 검증
            if not self._validate_dataset(df):
                raise ValueError("데이터셋 검증 실패")
            
            # 감성 분석 수행
            logger.info(f"감성 분석 시작 (모델: {models or 'all'})")
            results = []
            
            for idx, row in df.iterrows():
                try:
                    # content 컬럼 사용
                    text = str(row['content']).strip()
                    if not text:
                        logger.warning(f"빈 텍스트 발견 (index: {idx})")
                        continue
                        
                    # 텍스트 전처리
                    text = self.preprocess_text(text)
                    if not text:
                        logger.warning(f"전처리 후 빈 텍스트 (index: {idx})")
                        continue
                        
                    label, confidence = self.analyzer.predict(text)
                    
                    if confidence >= confidence_threshold:
                        results.append({
                            'video_id': row.get('video_id', f'video_{idx}'),
                            'content': text,
                            'sentiment': label,
                            'confidence': confidence,
                            'url': row.get('url', ''),
                            'title': row.get('title', ''),
                            'published_date': row.get('published_date', '')
                        })
                    else:
                        logger.debug(f"신뢰도 미달: {confidence:.3f} (index: {idx})")
                        
                except Exception as e:
                    logger.error(f"항목 분석 중 오류 발생 (index: {idx}): {str(e)}")
                    continue
            
            if not results:
                logger.warning("분석 결과가 없습니다")
                return pd.DataFrame()
            
            # 결과 데이터프레임 생성
            result_df = pd.DataFrame(results)
            
            # 결과 저장
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = output_path / f"sentiment_analysis_{timestamp}.csv"
                
                result_df.to_csv(output_file, index=False, encoding='utf-8')
                logger.info(f"분석 결과 저장 완료: {output_file}")
            
            # 감성 분포 로깅
            sentiment_dist = result_df['sentiment'].value_counts(normalize=True) * 100
            logger.info("감성 분포:")
            for sentiment, ratio in sentiment_dist.items():
                logger.info(f"  {sentiment}: {ratio:.1f}%")
            
            return result_df
            
        except Exception as e:
            logger.error(f"데이터셋 분석 중 오류 발생: {str(e)}")
            raise
    
    def validate_platform_data(self, df: pd.DataFrame, platform: str) -> Dict:
        """플랫폼별 데이터 검증"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # 필수 컬럼 검증
        required_columns = {
            'naver': ['title', 'content', 'published_date', 'platform', 'url'],
            'youtube': ['title', 'content', 'published_date', 'platform', 'url'],
            'google': ['title', 'content', 'published_date', 'platform', 'url']
        }
        
        if platform not in required_columns:
            validation_results['errors'].append(f"지원하지 않는 플랫폼: {platform}")
            validation_results['is_valid'] = False
            return validation_results
            
        # 필수 컬럼 검사
        missing_columns = [col for col in required_columns[platform] if col not in df.columns]
        if missing_columns:
            validation_results['errors'].append(f"필수 컬럼 누락: {', '.join(missing_columns)}")
            validation_results['is_valid'] = False
            
        # YouTube 데이터의 경우 video_id 처리
        if platform == 'youtube' and 'video_id' not in df.columns:
            logger.info("YouTube video_id 컬럼이 없어 URL에서 추출을 시도합니다.")
            try:
                # URL에서 video_id 추출 시도
                df['video_id'] = df['url'].apply(lambda x: self._extract_video_id(x) if isinstance(x, str) else None)
                
                # 추출 실패한 경우 임의의 ID 생성
                missing_ids = df['video_id'].isna()
                if missing_ids.any():
                    logger.warning(f"video_id 추출 실패: {missing_ids.sum()}개")
                    df.loc[missing_ids, 'video_id'] = [f"yt_{i:08d}" for i in range(missing_ids.sum())]
                
                validation_results['warnings'].append("video_id 컬럼이 자동 생성되었습니다.")
            except Exception as e:
                logger.error(f"video_id 생성 중 오류 발생: {str(e)}")
                validation_results['errors'].append("video_id 생성 실패")
                validation_results['is_valid'] = False
        
        # 날짜 형식 검증 및 변환
        if 'published_date' in df.columns:
            try:
                # 다양한 날짜 형식 처리 시도
                df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
                # YYYYMMDD 형식으로 변환
                df['published_date'] = df['published_date'].dt.strftime('%Y%m%d')
                # 변환 실패한 데이터 확인
                invalid_dates = df[df['published_date'].isna()]
                if not invalid_dates.empty:
                    validation_results['warnings'].append(f"잘못된 날짜 형식: {len(invalid_dates)}개")
                    # 변환 실패한 데이터는 None으로 유지 (시계열 분석에서 제외)
                    df.loc[invalid_dates.index, 'published_date'] = None
            except Exception as e:
                validation_results['warnings'].append(f"날짜 형식 변환 중 오류 발생: {str(e)}")
                
        # 내용 검증
        if 'content' in df.columns:
            empty_contents = df[df['content'].isna() | (df['content'].str.strip() == '')]
            if not empty_contents.empty:
                validation_results['warnings'].append(f"빈 내용: {len(empty_contents)}개")
                # 빈 내용은 제목으로 대체 (제목도 비어있는 경우 제외)
                for idx in empty_contents.index:
                    title = df.loc[idx, 'title']
                    if pd.notna(title) and title.strip():
                        df.loc[idx, 'content'] = title
                    else:
                        df.loc[idx, 'content'] = None  # 제목도 비어있는 경우 None으로 설정
                # None 값이 있는 행 제거
                df = df.dropna(subset=['content'])
                
        return validation_results

    def _extract_video_id(self, url: str) -> str:
        """YouTube URL에서 video_id 추출"""
        if not url or not isinstance(url, str):
            return ""
            
        # YouTube URL 패턴
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?]+)',  # 일반 URL
            r'youtube\.com\/embed\/([^&\n?]+)',  # 임베드 URL
            r'youtube\.com\/v\/([^&\n?]+)',  # 구버전 URL
            r'youtube\.com\/watch\?.*&v=([^&\n?]+)'  # 파라미터가 있는 URL
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        # URL이 아닌 경우 해시값 생성
        return f"video_{hash(url) & 0xFFFFFFFF:08x}"

    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        if not isinstance(text, str):
            return ""
            
        # 최대 길이 제한 (BERT 모델의 최대 토큰 수 고려)
        max_chars = 2048  # 한글 기준 대략적인 문자 수 (512 토큰 * 4)
        text = text[:max_chars]
        
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        # URL 제거
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        # 이메일 제거
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', ' ', text)
        # 특수문자 제거 (한글, 영문, 숫자, 기본 문장부호만 유지)
        text = re.sub(r'[^\w\s가-힣.,!?]', ' ', text)
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        # 앞뒤 공백 제거
        text = text.strip()
        
        return text

    def is_valid_for_sentiment(self, text: str) -> Tuple[bool, str]:
        """감성 분석 가능 여부 검증"""
        if not isinstance(text, str):
            return False, "텍스트가 문자열이 아님"
            
        # 기본 길이 검증
        if len(text) < self.sentiment_filters['min_text_length']:
            return False, f"텍스트가 너무 짧음 ({len(text)}자)"
        if len(text) > self.sentiment_filters['max_text_length']:
            return False, f"텍스트가 너무 김 ({len(text)}자)"
            
        # 토큰 수 예상 검증 (한글 기준 대략적인 계산)
        estimated_tokens = len(text) // 4  # 한글 1글자당 약 4토큰
        if estimated_tokens > 512:  # BERT 모델의 최대 토큰 수
            return False, f"예상 토큰 수 초과 ({estimated_tokens}토큰)"
            
        # 제외 키워드 검사
        for keyword in self.sentiment_filters['exclude_keywords']:
            if keyword in text.lower():
                return False, f"제외 키워드 포함: {keyword}"
                
        # 의미 있는 텍스트인지 검증 (최소 2개 이상의 단어)
        words = text.split()
        if len(words) < 2:
            return False, "의미 있는 단어가 부족함"
            
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
                try:
                    # predict 메서드가 있는 경우 사용, 없으면 analyze_text 사용
                    if hasattr(analyzer, 'predict'):
                        futures.append(executor.submit(analyzer.predict, text))
                    else:
                        futures.append(executor.submit(analyzer.analyze_text, text))
                except Exception as e:
                    logger.error(f"감성 분석 요청 중 오류: {str(e)}")
                    results.append((1, 0.0))  # 오류 시 중립으로 처리
            
            # 결과 수집
            for future in futures:
                try:
                    result = future.result()
                    if isinstance(result, tuple) and len(result) == 2:
                        results.append(result)
                    else:
                        logger.warning(f"잘못된 결과 형식: {result}")
                        results.append((1, 0.0))  # 잘못된 결과는 중립으로 처리
                except Exception as e:
                    logger.error(f"감성 분석 중 오류: {str(e)}")
                    results.append((1, 0.0))  # 오류 시 중립으로 처리
                    
        return results
    
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
            
            # 결과 검증 및 변환
            valid_results = []
            for result in sentiment_results:
                if isinstance(result, tuple) and len(result) == 2:
                    valid_results.append(result)
                else:
                    logger.warning(f"잘못된 결과 형식 무시: {result}")
                    valid_results.append((1, 0.0))  # 잘못된 결과는 중립으로 처리
            
            # 결과 추가
            sentiment_df = pd.DataFrame(valid_results, columns=['sentiment', 'confidence'])
            df = pd.concat([df, sentiment_df], axis=1)
            
            # 감성 레이블 추가
            df['sentiment_label'] = df['sentiment'].apply(lambda x: self.sentiment_map.get(x, 'neutral'))
            
            # 시각화를 위한 추가 컬럼
            df['sentiment_numeric'] = df['sentiment_label'].map(self.reverse_sentiment_map)
            df['sentiment_color'] = df['sentiment_label'].map({
                'positive': 'green',
                'neutral': 'gray',
                'negative': 'red'
            })
            
            # 신뢰도가 낮은 결과 필터링 (임계값 완화)
            df = df[df['confidence'] >= 0.2]  # 신뢰도 임계값을 0.2로 낮춤
            
            # 감성 분석 후 데이터가 비어있는지 확인
            if df.empty:
                logger.warning("감성 분석 후 데이터가 비어있습니다. 신뢰도 임계값을 확인해주세요.")
                return df
                
            logger.info(f"감성 분석 완료: {len(df)}개 항목 처리됨")
            
            # 감성 분포 로깅
            sentiment_dist = df['sentiment_label'].value_counts()
            logger.info("감성 분포:")
            for label, count in sentiment_dist.items():
                logger.info(f"- {label}: {count}개 ({count/len(df)*100:.1f}%)")
            
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

    def _validate_dataset(self, df: pd.DataFrame) -> bool:
        """데이터셋 유효성 검증"""
        try:
            if df is None or df.empty:
                logger.warning("데이터셋이 비어있습니다")
                return False

            # 필수 컬럼 검증
            required_columns = {
                'youtube': ['title', 'content', 'published_date', 'platform', 'url'],
                'naver': ['title', 'content', 'published_date', 'platform', 'url'],
                'google': ['title', 'content', 'published_date', 'platform', 'url']
            }

            # 플랫폼 확인
            platform = None
            if 'platform' in df.columns:
                platforms = df['platform'].unique()
                if len(platforms) == 1:
                    platform = platforms[0].lower()
                    if platform not in required_columns:
                        logger.warning(f"지원하지 않는 플랫폼: {platform}")
                        return False
                else:
                    logger.warning(f"여러 플랫폼이 혼합되어 있습니다: {platforms}")
                    # YouTube 데이터가 포함되어 있으면 YouTube 기준으로 검증
                    if 'youtube' in [p.lower() for p in platforms]:
                        platform = 'youtube'
                    else:
                        return False

            # 필수 컬럼 검사
            missing_columns = [col for col in required_columns[platform] if col not in df.columns]
            if missing_columns:
                logger.warning(f"필수 컬럼 누락: {missing_columns}")
                return False

            # YouTube 데이터의 경우 video_id 처리
            if platform == 'youtube' and 'video_id' not in df.columns:
                logger.info("YouTube video_id 컬럼이 없어 URL에서 추출을 시도합니다")
                try:
                    df['video_id'] = df['url'].apply(self._extract_video_id)
                    # 추출 실패한 경우 임의의 ID 생성
                    missing_ids = df['video_id'].isna()
                    if missing_ids.any():
                        logger.warning(f"video_id 추출 실패: {missing_ids.sum()}개")
                        df.loc[missing_ids, 'video_id'] = [f"yt_{i:08d}" for i in range(missing_ids.sum())]
                except Exception as e:
                    logger.error(f"video_id 생성 중 오류 발생: {str(e)}")
                    return False

            # 날짜 형식 검증
            if 'published_date' in df.columns:
                try:
                    df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
                    invalid_dates = df['published_date'].isna()
                    if invalid_dates.any():
                        logger.warning(f"날짜 형식 변환 실패: {invalid_dates.sum()}개")
                        # 변환 실패한 데이터는 현재 날짜로 대체
                        df.loc[invalid_dates.index, 'published_date'] = pd.Timestamp.now()
                except Exception as e:
                    logger.warning(f"날짜 형식 변환 중 오류 발생: {str(e)}")

            # 내용 검증
            if 'content' in df.columns:
                empty_contents = df[df['content'].isna() | (df['content'].str.strip() == '')]
                if not empty_contents.empty:
                    logger.warning(f"빈 내용: {len(empty_contents)}개")
                    # 빈 내용은 제목으로 대체
                    df.loc[empty_contents.index, 'content'] = df.loc[empty_contents.index, 'title']

            logger.info(f"데이터셋 검증 완료: {len(df)}개 항목")
            return True

        except Exception as e:
            logger.error(f"데이터셋 검증 중 오류 발생: {str(e)}")
            return False 