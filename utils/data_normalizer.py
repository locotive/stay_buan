import os
import json
import pandas as pd
from dateutil import parser
import logging
from typing import Dict
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class PlatformDataNormalizer:
    """플랫폼별 데이터 정규화 클래스"""
    
    def __init__(self):
        # 공통 필드 정의
        self.common_fields = {
            'platform': str,
            'title': str,
            'content': str,
            'published_date': str,
            'url': str,
            'sentiment': int,
            'confidence': float
        }
        
        # 플랫폼별 필드 매핑 정의
        self.platform_mappings = {
            'naver': {
                'title': 'title',
                'content': 'content',
                'published_date': 'pubDate',
                'url': 'url'
            },
            'youtube': {
                'title': 'title',
                'content': 'content',
                'published_date': 'published_date',
                'url': 'url'
            },
            'dcinside': {
                'title': 'subject',
                'content': 'memo',
                'published_date': 'date',
                'url': 'link'
            },
            'fmkorea': {
                'title': 'title',
                'content': 'content',
                'published_date': 'date',
                'url': 'url'
            },
            'buan': {
                'title': 'title',
                'content': 'content',
                'published_date': 'date',
                'url': 'url'
            }
        }

    def normalize_data(self, data, platform):
        """플랫폼별 데이터 정규화"""
        normalized_items = []
        
        for item in data:
            normalized_item = {}
            
            # 플랫폼 정보 추가
            normalized_item['platform'] = platform
            
            # 플랫폼별 매핑 적용
            mapping = self.platform_mappings.get(platform, {})
            for common_field, platform_field in mapping.items():
                value = self._get_nested_value(item, platform_field)
                normalized_item[common_field] = value
            
            # 날짜 형식 통일
            if 'published_date' in normalized_item:
                normalized_item['published_date'] = self._normalize_date(
                    normalized_item['published_date'],
                    platform
                )
            
            # URL 형식 통일
            if 'url' in normalized_item:
                normalized_item['url'] = self._normalize_url(
                    normalized_item['url'],
                    platform
                )
            
            normalized_items.append(normalized_item)
        
        return normalized_items

    def _get_nested_value(self, item, field_path):
        """중첩된 필드 값 추출"""
        if not isinstance(item, dict):
            return ''
            
        if '.' not in field_path:
            value = item.get(field_path, '')
            return str(value) if value is not None else ''
            
        parts = field_path.split('.')
        value = item
        
        for part in parts:
            if not isinstance(value, dict):
                return ''
            value = value.get(part)
            if value is None:
                return ''
                
        return str(value) if value is not None else ''

    def _normalize_date(self, date_str, platform):
        """날짜 형식 정규화"""
        if not date_str or not isinstance(date_str, str):
            return None
        
        try:
            # 잘못된 형식의 날짜 문자열 처리 (예: "12202517" -> "20251217")
            if len(date_str) == 8 and date_str.isdigit():
                # 연도가 2자리로 시작하는 경우 (예: 12xxxxxx)
                if date_str.startswith(('12', '13', '14')):
                    year = '20' + date_str[:2]
                    month = date_str[2:4]
                    day = date_str[4:6]
                    date_str = f"{year}{month}{day}"
            
            # 다양한 날짜 형식 처리
            date_formats = [
                '%Y%m%d',  # 20250513
                '%Y-%m-%d',  # 2025-05-13
                '%Y.%m.%d',  # 2025.05.13
                '%Y년%m월%d일',  # 2025년05월13일
                '%Y/%m/%d',  # 2025/05/13
                '%d/%m/%Y',  # 13/05/2025
                '%m/%d/%Y',  # 05/13/2025
            ]
            
            # 플랫폼별 특수 날짜 형식 처리
            if platform == 'naver_news_api':
                if date_str.startswith(('12', '13', '14')):
                    year = '20' + date_str[:2]
                    month = date_str[2:4]
                    day = date_str[4:6]
                    date_str = f"{year}{month}{day}"
            
            # 날짜 파싱 시도
            parsed_date = None
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
                
            if parsed_date is None:
                return None
            
            # 유효한 날짜 범위 체크 (2000년 ~ 현재)
            current_year = datetime.now().year
            if parsed_date.year < 2000 or parsed_date.year > current_year:
                return None
            
            # YYYYMMDD 형식으로 반환
            return parsed_date.strftime('%Y%m%d')
        
        except Exception as e:
            logger.warning(f"날짜 정규화 중 오류 발생: {str(e)}, 원본: {date_str}")
            return None

    def _normalize_url(self, url, platform):
        """URL 형식 통일"""
        if platform == 'youtube':
            return f"https://www.youtube.com/watch?v={url}"
        return url

class DataValidator:
    """데이터 검증 클래스"""
    
    def validate_data(self, df):
        """데이터 유효성 검사"""
        validation_results = {
            'missing_fields': [],
            'invalid_dates': [],
            'empty_content': []
        }
        
        required_fields = ['platform', 'title', 'content', 'published_date']
        for field in required_fields:
            if field not in df.columns:
                validation_results['missing_fields'].append(field)
        
        if 'published_date' in df.columns:
            invalid_dates = df[~df['published_date'].astype(str).str.match(r'^\d{8}$')]
            if not invalid_dates.empty:
                normalizer = PlatformDataNormalizer()
                df.loc[invalid_dates.index, 'published_date'] = df.loc[invalid_dates.index, 'published_date'].apply(
                    lambda x: normalizer._normalize_date(x, df['platform'].iloc[0])
                )
                validation_results['invalid_dates'].extend(invalid_dates.index.tolist())
        
        if 'content' in df.columns:
            empty_content = df[df['content'].str.strip() == '']
            if not empty_content.empty:
                validation_results['empty_content'].extend(empty_content.index.tolist())
        
        return validation_results

def load_and_normalize_data(filepath):
    """데이터 로드 및 정규화"""
    try:
        platform = os.path.basename(filepath).split('_')[0]
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        normalizer = PlatformDataNormalizer()
        normalized_data = normalizer.normalize_data(raw_data, platform)
        df = pd.DataFrame(normalized_data)
        validator = DataValidator()
        validation_results = validator.validate_data(df)
        if any(validation_results.values()):
            logger.warning(f"데이터 검증 결과: {validation_results}")
        return df
    except Exception as e:
        logger.error(f"데이터 로드 및 정규화 중 오류 발생: {str(e)}")
        raise
