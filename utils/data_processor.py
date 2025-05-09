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
        self.sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.cache_dir = "data/cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 감성 분석 필터링 조건
        self.sentiment_filters = {
            'min_text_length': 10,  # 최소 텍스트 길이
            'max_text_length': 1000,  # 최대 텍스트 길이
            'min_confidence': 0.6,  # 최소 신뢰도
            'exclude_keywords': ['광고', '홍보', 'sponsored'],  # 제외할 키워드
            'required_keywords': ['부안']  # 필수 포함 키워드
        }
    
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
        
        # 특수문자 제거
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
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
            invalid_dates = df[~df['published_date'].str.match(r'^\d{8}$')]
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
        try:
            # 1. 데이터 검증
            validation_results = self.validate_platform_data(df, platform)
            if not validation_results['is_valid']:
                raise ValueError(f"데이터 검증 실패: {validation_results['errors']}")
            
            # 2. 텍스트 전처리
            if 'content' in df.columns:
                df['content'] = df['content'].apply(self.preprocess_text)
            if 'title' in df.columns:
                df['title'] = df['title'].apply(self.preprocess_text)
            
            # 3. 감성 분석
            if analyzer is not None:
                # 배치 단위 감성 분석
                sentiment_results = self.analyze_sentiment_batch(df['content'].tolist(), analyzer)
                
                # 결과 추가
                sentiment_df = pd.DataFrame(sentiment_results, columns=['sentiment', 'confidence'])
                df = pd.concat([df, sentiment_df], axis=1)
                
                # 감성 레이블 추가
                df['sentiment_label'] = df['sentiment'].map(self.sentiment_map)
                
                # 신뢰도가 낮은 결과 필터링
                df = df[df['confidence'] >= self.sentiment_filters['min_confidence']]
            
            # 4. 날짜 정규화
            if 'published_date' in df.columns:
                df['published_date'] = pd.to_datetime(df['published_date'], format='%Y%m%d', errors='coerce')
            
            # 5. 중복 제거
            df = df.drop_duplicates(subset=['title', 'content', 'published_date'])
            
            return df
            
        except Exception as e:
            logger.error(f"데이터 처리 중 오류 발생: {str(e)}")
            raise
    
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