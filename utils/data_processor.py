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
import time
import glob

logger = logging.getLogger(__name__)

class SentimentAnalysisHistory:
    """감성 분석 이력 관리 클래스"""
    
    def __init__(self, history_file="data/logs/sentiment_analysis_history.json"):
        self.history_file = history_file
        self.history = self._load_history()
        
    def _load_history(self):
        """이력 파일 로드"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"이력 파일 로드 실패: {str(e)}")
                return {"analyses": []}
        return {"analyses": []}
    
    def _save_history(self):
        """이력 파일 저장"""
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"이력 파일 저장 실패: {str(e)}")
    
    def add_analysis(self, file_path, models, item_count, sentiment_distribution, processing_time):
        """새로운 분석 이력 추가"""
        analysis = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_path": file_path,
            "models": models,
            "item_count": item_count,
            "sentiment_distribution": sentiment_distribution,
            "processing_time": processing_time
        }
        self.history["analyses"].append(analysis)
        self._save_history()
    
    def get_file_analysis_history(self, file_path):
        """특정 파일의 분석 이력 조회"""
        return [a for a in self.history["analyses"] if a["file_path"] == file_path]
    
    def is_file_analyzed(self, file_path, models=None):
        """파일이 이미 분석되었는지 확인"""
        file_history = self.get_file_analysis_history(file_path)
        if not file_history:
            return False
        if models is None:
            return True
        # 모델 목록이 지정된 경우, 해당 모델들로 분석된 이력이 있는지 확인
        return any(set(models) == set(a["models"]) for a in file_history)

class DataProcessor:
    """데이터 전처리 파이프라인 클래스"""
    
    def __init__(self):
        self.sentiment_map = {
            0: 'negative',
            1: 'neutral',
            2: 'positive'
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
            'kcelectra-base-v2022': {
                'name': 'beomi/kcelectra-base-v2022',
                'local_path': os.path.join(self.model_dir, 'kcelectra-base-v2022'),
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
            'max_text_length': 4096,  # 최대 길이 증가
            'min_confidence': 0.2,
            'exclude_keywords': ['광고', '홍보', 'sponsored'],
            'required_keywords': [],
            'min_content_words': 3,
            'max_tokens': 512,  # BERT 모델의 최대 토큰 수
            'chunk_size': 1024  # 청크 크기 설정
        }
        
        # 사용 가능한 감성분석 모델 목록
        self.available_models = {
            'kobert': 'KoBERT (가벼운 모델)',
            'kcelectra-base-v2022': 'KcELECTRA-base-v2022 (감성분석 특화)',
            'kcelectra': 'KcELECTRA (감성분석 특화)',
            'kcbert-large': 'KcBERT-large (큰 모델)',
            'kosentencebert': 'KoSentenceBERT (문장 수준 분석)'
        }
        
        # 미리 정의된 모델 조합
        self.model_combinations = {
            'light': ['kobert', 'kcelectra-base-v2022'],  # 가벼운 조합
            'balanced': ['kcelectra-base-v2022', 'kcelectra', 'kosentencebert'],  # 균형잡힌 조합
            'heavy': ['kcbert-large', 'kosentencebert', 'kcelectra-base-v2022'],  # 정확도 중심
            'custom': []  # 사용자 정의
        }
        
        # 모델 초기화 상태
        self._initialized_models = {}
        
        self.analyzer = EnsembleSentimentAnalyzer()
        self.sentiment_history = SentimentAnalysisHistory()
    
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
                model_names = ['kobert', 'kcelectra']
            
            # 선택된 모델들 초기화
            models = [self._initialize_model(name) for name in model_names]
            
            return EnsembleSentimentAnalyzer(models=models)
                
        except Exception as e:
            logger.error(f"감성분석기 초기화 중 오류 발생: {str(e)}")
            raise
    
    def analyze_dataset(self, input_file, models=None, output_dir="data/processed", progress_callback=None):
        """데이터셋 감성 분석"""
        try:
            start_time = time.time()
            
            # 데이터 로드 및 검증
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 빈 내용 확인
            empty_contents = sum(1 for item in data if not item.get('content', '').strip())
            if empty_contents > 0:
                logger.warning(f"빈 내용: {empty_contents}개")
            
            # 데이터 검증
            valid_data = [item for item in data if item.get('content', '').strip()]
            logger.info(f"데이터셋 검증 완료: {len(valid_data)}개 항목")
            
            if not valid_data:
                logger.error("분석할 데이터가 없습니다.")
                return None
            
            # 감성 분석 시작
            logger.info(f"감성 분석 시작 (모델: {models})")
            
            # 앙상블 분석기 초기화
            analyzer = EnsembleSentimentAnalyzer(models=models)
            
            # 결과 저장을 위한 리스트
            results = []
            
            # 각 항목에 대해 감성 분석 수행
            for i, item in enumerate(valid_data):
                try:
                    # 텍스트 전처리
                    text = self.preprocess_text(item['content'])
                    if not text:
                        logger.warning(f"전처리 후 빈 텍스트 (index: {i})")
                        continue
                    
                    # 감성 분석 수행
                    sentiment, confidence = analyzer.predict(text)
                    logger.info(f"감성 분석 결과: {sentiment}, 신뢰도: {confidence:.3f}")
                    # 결과 저장
                    result = {
                        'title': item.get('title', ''),
                        'content': text,
                        'platform': item.get('platform', 'unknown'),
                        'published_date': item.get('published_date', ''),
                        'url': item.get('url', ''),
                        'sentiment': sentiment,
                        'confidence': confidence
                    }
                    results.append(result)
                    
                    # 진행 상황 업데이트
                    if progress_callback:
                        progress_callback(i + 1)
                    
                except Exception as e:
                    logger.error(f"항목 {i} 분석 중 오류: {str(e)}")
                    continue
            
            # 결과를 DataFrame으로 변환
            df = pd.DataFrame(results)
            
            # 감성 분포 계산
            if not df.empty:
                sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100
                logger.info("감성 분포:")
                for sentiment, percentage in sentiment_counts.items():
                    sentiment_label = 'negative' if sentiment == 0 else 'neutral' if sentiment == 1 else 'positive'
                    logger.info(f"  {sentiment_label}: {percentage:.1f}%")
            
            # 결과 저장 부분 수정
            processing_time = time.time() - start_time
            save_results = self.save_analysis_results(
                df=df,
                input_file=input_file,
                models=models,
                processing_time=processing_time,
                output_dir=output_dir
            )
            
            logger.info(f"분석 결과 저장 완료: {save_results['csv_path']}")
            return df
            
        except Exception as e:
            logger.error(f"데이터셋 분석 중 오류 발생: {str(e)}")
            return None
    
    def validate_platform_data(self, df: pd.DataFrame, platform: str) -> Dict:
        """플랫폼별 데이터 검증"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # 필수 컬럼 검증
        required_columns = {
            # 네이버 플랫폼
            'naver': ['title', 'content', 'published_date', 'platform', 'url'],
            'naver_blog': ['title', 'content', 'published_date', 'platform', 'url'],
            'naver_blog_api': ['title', 'content', 'published_date', 'platform', 'url'],
            'naver_cafe': ['title', 'content', 'published_date', 'platform', 'url'],
            'naver_cafearticle': ['title', 'content', 'published_date', 'platform', 'url'],
            'naver_cafearticle_api': ['title', 'content', 'published_date', 'platform', 'url'],
            'naver_news': ['title', 'content', 'published_date', 'platform', 'url'],
            'naver_community': ['title', 'content', 'published_date', 'platform', 'url'],
            
            # 유튜브 플랫폼
            'youtube': ['title', 'content', 'published_date', 'platform', 'url', 'video_id'],
            'youtube_api': ['title', 'content', 'published_date', 'platform', 'url', 'video_id'],
            
            # 구글 플랫폼
            'google': ['title', 'content', 'published_date', 'platform', 'url'],
            'google_search': ['title', 'content', 'published_date', 'platform', 'url'],
            'google_news': ['title', 'content', 'published_date', 'platform', 'url'],
            
            # 부안군청 플랫폼
            'buan_gov': ['title', 'content', 'published_date', 'platform', 'url'],
            'buan_gov_news': ['title', 'content', 'published_date', 'platform', 'url'],
            'buan_gov_notice': ['title', 'content', 'published_date', 'platform', 'url'],
            
            # 커뮤니티 플랫폼
            'community': ['title', 'content', 'published_date', 'platform', 'url'],
            'local_community': ['title', 'content', 'published_date', 'platform', 'url'],
            'tour_community': ['title', 'content', 'published_date', 'platform', 'url']
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
        """텍스트 전처리 개선"""
        if not isinstance(text, str):
            return ""
            
        # 1. 기본 전처리
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
        
        # 2. 길이 제한 및 청크 처리
        if len(text) > self.sentiment_filters['max_text_length']:
            # 문장 단위로 분할
            sentences = re.split(r'[.!?]+', text)
            chunks = []
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # 현재 문장의 예상 토큰 수 계산 (한글 기준)
                sentence_tokens = len(sentence) // 4
                
                if current_length + sentence_tokens > self.sentiment_filters['max_tokens']:
                    # 현재 청크가 최소 길이를 만족하면 저장
                    if current_length >= self.sentiment_filters['min_text_length']:
                        chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_tokens
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_tokens
            
            # 마지막 청크 처리
            if current_chunk and current_length >= self.sentiment_filters['min_text_length']:
                chunks.append(' '.join(current_chunk))
            
            # 가장 의미 있는 청크 선택 (첫 번째 청크 사용)
            if chunks:
                text = chunks[0]
            else:
                # 청크가 없는 경우 앞부분만 사용
                text = text[:self.sentiment_filters['max_text_length']]
        
        return text

    def is_valid_for_sentiment(self, text: str) -> Tuple[bool, str]:
        """감성 분석 가능 여부 검증 개선"""
        if not isinstance(text, str):
            return False, "텍스트가 문자열이 아님"
            
        # 기본 길이 검증
        if len(text) < self.sentiment_filters['min_text_length']:
            return False, f"텍스트가 너무 짧음 ({len(text)}자)"
            
        # 토큰 수 예상 검증 (한글 기준 대략적인 계산)
        estimated_tokens = len(text) // 4  # 한글 1글자당 약 4토큰
        if estimated_tokens > self.sentiment_filters['max_tokens']:
            # 긴 텍스트는 자동으로 전처리되므로 경고만 표시
            logger.warning(f"긴 텍스트 감지: {estimated_tokens}토큰 (자동 전처리 적용)")
            
        # 제외 키워드 검사
        for keyword in self.sentiment_filters['exclude_keywords']:
            if keyword in text.lower():
                return False, f"제외 키워드 포함: {keyword}"
                
        # 의미 있는 텍스트인지 검증 (최소 2개 이상의 단어)
        words = text.split()
        if len(words) < 2:
            return False, "의미 있는 단어가 부족함"
            
        return True, "유효한 텍스트"
    
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
                    'excluded_keywords': self.sentiment_filters['exclude_keywords']
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
                # 네이버 플랫폼
                'naver': ['title', 'content', 'published_date', 'platform', 'url'],
                'naver_blog': ['title', 'content', 'published_date', 'platform', 'url'],
                'naver_blog_api': ['title', 'content', 'published_date', 'platform', 'url'],
                'naver_cafe': ['title', 'content', 'published_date', 'platform', 'url'],
                'naver_cafearticle': ['title', 'content', 'published_date', 'platform', 'url'],
                'naver_cafearticle_api': ['title', 'content', 'published_date', 'platform', 'url'],
                'naver_news': ['title', 'content', 'published_date', 'platform', 'url'],
                'naver_community': ['title', 'content', 'published_date', 'platform', 'url'],
                
                # 유튜브 플랫폼
                'youtube': ['title', 'content', 'published_date', 'platform', 'url'],
                'youtube_api': ['title', 'content', 'published_date', 'platform', 'url'],
                
                # 구글 플랫폼
                'google': ['title', 'content', 'published_date', 'platform', 'url'],
                'google_search': ['title', 'content', 'published_date', 'platform', 'url'],
                'google_news': ['title', 'content', 'published_date', 'platform', 'url'],
                
                # 부안군청 플랫폼
                'buan_gov': ['title', 'content', 'published_date', 'platform', 'url'],
                'buan_gov_news': ['title', 'content', 'published_date', 'platform', 'url'],
                'buan_gov_notice': ['title', 'content', 'published_date', 'platform', 'url'],
                
                # 커뮤니티 플랫폼
                'community': ['title', 'content', 'published_date', 'platform', 'url'],
                'local_community': ['title', 'content', 'published_date', 'platform', 'url'],
                'tour_community': ['title', 'content', 'published_date', 'platform', 'url']
            }

            # 플랫폼 확인
            platform = None
            if 'platform' in df.columns:
                platforms = df['platform'].unique()
                if len(platforms) == 1:
                    platform = platforms[0].lower()
                    
                    # 플랫폼 매핑
                    platform_mapping = {
                        # 네이버 플랫폼 - naver_로 시작하는 모든 플랫폼을 naver로 매핑
                        'naver_blog': 'naver',
                        'naver_blog_api': 'naver',
                        'naver_cafe': 'naver',
                        'naver_cafearticle': 'naver',
                        'naver_cafearticle_api': 'naver',
                        'naver_news': 'naver',
                        'naver_community': 'naver',
                        
                        # 유튜브 플랫폼
                        'youtube_api': 'youtube',
                        
                        # 구글 플랫폼
                        'google_search': 'google',
                        'google_news': 'google',
                        
                        # 부안군청 플랫폼
                        'buan_gov_news': 'buan_gov',
                        'buan_gov_notice': 'buan_gov',
                        
                        # 커뮤니티 플랫폼
                        'local_community': 'community',
                        'tour_community': 'community'
                    }
                    
                    # 매핑된 플랫폼으로 변경
                    if platform in platform_mapping:
                        platform = platform_mapping[platform]
                        df['platform'] = platform
                    elif platform.startswith('naver_'):
                        # naver_로 시작하는 모든 플랫폼을 naver로 매핑
                        platform = 'naver'
                        df['platform'] = platform
                    elif platform not in ['naver', 'youtube', 'google', 'buan_gov', 'community']:
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
            if platform in required_columns:
                missing_columns = [col for col in required_columns[platform] if col not in df.columns]
                if missing_columns:
                    # video_id가 누락된 경우 자동 생성 시도
                    if platform == 'youtube' and 'video_id' in missing_columns:
                        logger.info("YouTube video_id 컬럼이 없어 URL에서 추출을 시도합니다")
                        try:
                            df['video_id'] = df['url'].apply(self._extract_video_id)
                            # 추출 실패한 경우 임의의 ID 생성
                            missing_ids = df['video_id'].isna()
                            if missing_ids.any():
                                logger.warning(f"video_id 추출 실패: {missing_ids.sum()}개")
                                df.loc[missing_ids, 'video_id'] = [f"yt_{i:08d}" for i in range(missing_ids.sum())]
                            missing_columns.remove('video_id')
                        except Exception as e:
                            logger.error(f"video_id 생성 중 오류 발생: {str(e)}")
                    
                    if missing_columns:
                        logger.warning(f"필수 컬럼 누락: {missing_columns}")
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

    def get_analysis_history(self, file_path=None):
        """감성 분석 이력 조회"""
        if file_path:
            return self.sentiment_history.get_file_analysis_history(file_path)
        return self.sentiment_history.history["analyses"] 

    def save_analysis_results(self, df: pd.DataFrame, input_file: str, models: List[str], processing_time: float, output_dir: str = "data/processed") -> Dict[str, str]:
        """감성 분석 결과 저장 (CSV + JSON)"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"sentiment_analysis_{timestamp}"
            
            # CSV 파일 저장
            csv_path = os.path.join(output_dir, f"{base_filename}.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8')
            
            # 감성 분포 계산
            sentiment_distribution = {
                'negative': f"{len(df[df['sentiment'] == 'negative']) / len(df) * 100:.1f}%",
                'neutral': f"{len(df[df['sentiment'] == 'neutral']) / len(df) * 100:.1f}%",
                'positive': f"{len(df[df['sentiment'] == 'positive']) / len(df) * 100:.1f}%"
            }
            
            # JSON 메타데이터 생성
            metadata = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'input_file': input_file,
                'models': models,
                'item_count': len(df),
                'processing_time': processing_time,
                'sentiment_distribution': sentiment_distribution,
                'output_files': {
                    'csv': csv_path,
                    'json': os.path.join(output_dir, f"{base_filename}.json")
                },
                'analysis_parameters': {
                    'min_text_length': self.sentiment_filters['min_text_length'],
                    'max_text_length': self.sentiment_filters['max_text_length'],
                    'min_confidence': self.sentiment_filters['min_confidence'],
                    'excluded_keywords': self.sentiment_filters['exclude_keywords']
                }
            }
            
            # JSON 파일 저장
            json_path = os.path.join(output_dir, f"{base_filename}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # 분석 이력에 추가
            self.sentiment_history.add_analysis(
                file_path=input_file,
                models=models,
                item_count=len(df),
                sentiment_distribution=sentiment_distribution,
                processing_time=processing_time
            )
            
            logger.info(f"감성 분석 결과 저장 완료: {csv_path}, {json_path}")
            return {
                'csv_path': csv_path,
                'json_path': json_path,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"결과 저장 중 오류 발생: {str(e)}")
            raise 

    def validate_and_transform_data(self, df):
        """데이터 유효성 검사 및 변환"""
        if df.empty:
            logger.warning("입력 데이터프레임이 비어있습니다.")
            return pd.DataFrame()
        
        try:
            # 필수 컬럼 확인
            required_columns = ['title', 'content', 'published_date', 'platform']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"필수 컬럼이 누락되었습니다: {missing_columns}")
                return pd.DataFrame()
            
            # 날짜 데이터 처리
            def normalize_date(date_str):
                logger.info(f"날짜 변환 시작 - 입력값: {date_str}, 타입: {type(date_str)}")
                
                if pd.isna(date_str):
                    logger.warning("날짜값이 NA입니다.")
                    return None
                    
                try:
                    # 문자열로 변환
                    date_str = str(date_str).strip()
                    logger.info(f"문자열 변환 후: {date_str}")
                    
                    # 이미 YYYYMMDD 형식인 경우
                    if re.match(r'^\d{8}$', date_str):
                        logger.info(f"YYYYMMDD 형식 감지: {date_str}")
                        return date_str
                    
                    # 날짜 형식 변환 시도
                    try:
                        logger.info(f"pandas to_datetime 시도: {date_str}")
                        dt = pd.to_datetime(date_str)
                        result = dt.strftime('%Y%m%d')
                        logger.info(f"변환 성공: {date_str} -> {result}")
                        return result
                    except Exception as e:
                        logger.warning(f"pandas to_datetime 실패: {str(e)}")
                        
                        # 한글 날짜 형식 처리
                        if '년' in date_str and '월' in date_str and '일' in date_str:
                            logger.info(f"한글 날짜 형식 감지: {date_str}")
                            try:
                                normalized = date_str.replace('년', '-').replace('월', '-').replace('일', '')
                                logger.info(f"한글 날짜 정규화: {normalized}")
                                dt = pd.to_datetime(normalized)
                                result = dt.strftime('%Y%m%d')
                                logger.info(f"한글 날짜 변환 성공: {date_str} -> {result}")
                                return result
                            except Exception as e:
                                logger.error(f"한글 날짜 변환 실패: {str(e)}")
                        
                        # 다른 형식 시도
                        try:
                            # YYYY-MM-DD 형식으로 변환 시도
                            if re.match(r'\d{4}[-./]\d{1,2}[-./]\d{1,2}', date_str):
                                logger.info(f"구분자 포함 날짜 형식 감지: {date_str}")
                                normalized = re.sub(r'[-./]', '-', date_str)
                                dt = pd.to_datetime(normalized)
                                result = dt.strftime('%Y%m%d')
                                logger.info(f"구분자 날짜 변환 성공: {date_str} -> {result}")
                                return result
                        except Exception as e:
                            logger.warning(f"구분자 날짜 변환 실패: {str(e)}")
                        
                        logger.warning(f"모든 날짜 변환 시도 실패: {date_str}")
                        return None
                        
                except Exception as e:
                    logger.error(f"날짜 변환 중 예외 발생: {str(e)}, 입력값: {date_str}")
                    return None
            
            # 날짜 변환 적용 전 로깅
            logger.info("=== 날짜 변환 시작 ===")
            logger.info(f"총 {len(df)}개의 날짜 데이터 처리 시작")
            logger.info(f"날짜 데이터 샘플:\n{df['published_date'].head()}")
            
            # 날짜 변환 적용
            df['published_date'] = df['published_date'].apply(normalize_date)
            
            # 변환 결과 로깅
            invalid_dates = df['published_date'].isna()
            logger.info(f"=== 날짜 변환 결과 ===")
            logger.info(f"총 데이터 수: {len(df)}")
            logger.info(f"유효한 날짜 수: {len(df) - invalid_dates.sum()}")
            logger.info(f"유효하지 않은 날짜 수: {invalid_dates.sum()}")
            if invalid_dates.any():
                logger.warning("유효하지 않은 날짜 샘플:")
                invalid_samples = df[invalid_dates]['published_date'].head()
                logger.warning(f"\n{invalid_samples}")
            
            # 유효하지 않은 날짜 필터링
            if invalid_dates.any():
                logger.warning(f"유효하지 않은 날짜 데이터 {invalid_dates.sum()}개 발견")
                df = df[~invalid_dates]
            
            # 현재 날짜보다 미래 날짜 필터링
            current_date = pd.Timestamp.now().strftime('%Y%m%d')
            future_dates = df['published_date'] > current_date
            if future_dates.any():
                logger.warning(f"미래 날짜 데이터 {future_dates.sum()}개 발견")
                logger.warning("미래 날짜 샘플:")
                future_samples = df[future_dates]['published_date'].head()
                logger.warning(f"\n{future_samples}")
                df = df[~future_dates]
            
            # 2000년 이전 데이터 필터링
            old_dates = df['published_date'].str[:4].astype(int) < 2000
            if old_dates.any():
                logger.warning(f"2000년 이전 데이터 {old_dates.sum()}개 발견")
                logger.warning("2000년 이전 날짜 샘플:")
                old_samples = df[old_dates]['published_date'].head()
                logger.warning(f"\n{old_samples}")
                df = df[~old_dates]
            
            # 최종 결과 로깅
            logger.info("=== 최종 날짜 처리 결과 ===")
            logger.info(f"최종 데이터 수: {len(df)}")
            logger.info("날짜 범위:")
            if not df.empty:
                logger.info(f"시작: {df['published_date'].min()}")
                logger.info(f"종료: {df['published_date'].max()}")
            
            # 제목과 내용 처리
            df['title'] = df['title'].fillna('').astype(str)
            df['content'] = df['content'].fillna('').astype(str)
            
            # 내용이 비어있는 경우 제목으로 대체
            empty_content = df['content'].str.strip() == ''
            if empty_content.any():
                logger.warning(f"내용이 비어있는 데이터 {empty_content.sum()}개 발견")
                df.loc[empty_content, 'content'] = df.loc[empty_content, 'title']
            
            # 중복 데이터 제거
            initial_len = len(df)
            df = df.drop_duplicates(subset=['title', 'content', 'published_date'])
            if len(df) < initial_len:
                logger.info(f"중복 데이터 {initial_len - len(df)}개 제거됨")
            
            return df
        
        except Exception as e:
            logger.error(f"데이터 처리 중 오류 발생: {str(e)}")
            return pd.DataFrame() 

def load_combined_data():
    """최근 크롤링된 데이터를 모두 로드하여 통합"""
    try:
        # 최근 combined 파일 찾기
        combined_files = glob.glob("data/raw/combined_*.json")
        if not combined_files:
            logger.warning("통합 데이터 파일을 찾을 수 없습니다.")
            return pd.DataFrame()
        
        # 가장 최근 파일 선택
        latest_file = max(combined_files, key=os.path.getmtime)
        
        # JSON 파일 로드
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # DataFrame으로 변환
        df = pd.DataFrame(data)
        
        # 날짜 형식 변환
        if 'published_date' in df.columns:
            df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
        
        return df
    
    except Exception as e:
        logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
        return pd.DataFrame()

def get_platform_stats(df):
    """플랫폼별 게시물 수 통계"""
    if df.empty:
        return pd.DataFrame({'platform': [], 'count': []})
    
    platform_stats = df['platform'].value_counts().reset_index()
    platform_stats.columns = ['platform', 'count']
    return platform_stats

def get_keyword_stats(df):
    """키워드별 게시물 수 통계"""
    if df.empty:
        return pd.DataFrame({'keyword': [], 'count': []})
    
    # 키워드 컬럼이 있는 경우
    if 'keywords' in df.columns:
        # 키워드 문자열을 리스트로 변환
        df['keyword_list'] = df['keywords'].apply(lambda x: x.split() if isinstance(x, str) else [])
        
        # 모든 키워드 카운트
        keyword_counts = {}
        for keywords in df['keyword_list']:
            for keyword in keywords:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        # DataFrame으로 변환
        keyword_stats = pd.DataFrame([
            {'keyword': k, 'count': v} for k, v in keyword_counts.items()
        ])
        
        return keyword_stats.sort_values('count', ascending=False)
    
    return pd.DataFrame({'keyword': [], 'count': []})

def get_trend_data(df):
    """시간대별 게시물 수 트렌드"""
    if df.empty or 'published_date' not in df.columns:
        return pd.DataFrame({'hour': [], 'count': []})
    
    # 시간대별 카운트
    df['hour'] = df['published_date'].dt.hour
    trend_data = df['hour'].value_counts().reset_index()
    trend_data.columns = ['hour', 'count']
    
    # 시간순 정렬
    trend_data = trend_data.sort_values('hour')
    
    return trend_data

def get_recent_articles(df, n=10):
    """최근 게시물 목록"""
    if df.empty:
        return pd.DataFrame()
    
    # 필요한 컬럼만 선택
    columns = ['title', 'platform', 'published_date', 'url']
    available_columns = [col for col in columns if col in df.columns]
    
    # 최근 게시물 선택
    recent_df = df[available_columns].copy()
    
    # 날짜순 정렬
    if 'published_date' in recent_df.columns:
        recent_df = recent_df.sort_values('published_date', ascending=False)
    
    # 상위 n개 선택
    recent_df = recent_df.head(n)
    
    # 날짜 형식 변환
    if 'published_date' in recent_df.columns:
        recent_df['published_date'] = recent_df['published_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return recent_df

def analyze_sentiment(text, analyzer):
    """텍스트 감성 분석"""
    try:
        sentiment, confidence = analyzer.analyze(text)
        return sentiment, confidence
    except Exception as e:
        logger.error(f"감성 분석 중 오류 발생: {str(e)}")
        return 'unknown', 0.0

def clear_analysis_cache():
    """분석 캐시 초기화"""
    try:
        cache_dir = "data/cache"
        if os.path.exists(cache_dir):
            for file in glob.glob(os.path.join(cache_dir, "*.cache")):
                os.remove(file)
        return True
    except Exception as e:
        logger.error(f"캐시 초기화 중 오류 발생: {str(e)}")
        return False 