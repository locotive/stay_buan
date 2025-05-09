import os
import json
import pandas as pd
from dateutil import parser
import logging

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
                'content': 'description',
                'published_date': 'pubDate',
                'url': 'link'
            },
            'youtube': {
                'title': 'snippet.title',
                'content': 'snippet.description',
                'published_date': 'snippet.publishedAt',
                'url': 'id.videoId'
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
        if '.' not in field_path:
            return item.get(field_path, '')
            
        parts = field_path.split('.')
        value = item
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part, '')
            else:
                return ''
        return value
    
    def _normalize_date(self, date_str, platform):
        """날짜 형식 통일"""
        try:
            if platform == 'youtube':
                # YouTube: "2024-03-15T10:30:00Z" -> "20240315"
                return date_str.split('T')[0].replace('-', '')
            elif platform == 'naver':
                # Naver: "Wed, 15 Mar 2024 10:30:00 +0900" -> "20240315"
                return parser.parse(date_str).strftime('%Y%m%d')
            else:
                # 기타 플랫폼: "2024-03-15" -> "20240315"
                return date_str.replace('-', '')
        except:
            return ''
    
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
        
        # 필수 필드 확인
        required_fields = ['platform', 'title', 'content', 'published_date']
        for field in required_fields:
            if field not in df.columns:
                validation_results['missing_fields'].append(field)
        
        # 날짜 형식 검증
        if 'published_date' in df.columns:
            invalid_dates = df[~df['published_date'].str.match(r'^\d{8}$')]
            if not invalid_dates.empty:
                validation_results['invalid_dates'].extend(invalid_dates.index.tolist())
        
        # 빈 컨텐츠 확인
        if 'content' in df.columns:
            empty_content = df[df['content'].str.strip() == '']
            if not empty_content.empty:
                validation_results['empty_content'].extend(empty_content.index.tolist())
        
        return validation_results

def load_and_normalize_data(filepath):
    """데이터 로드 및 정규화"""
    try:
        # 파일명에서 플랫폼 추출
        platform = os.path.basename(filepath).split('_')[0]
        
        # 데이터 로드
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 데이터 정규화
        normalizer = PlatformDataNormalizer()
        normalized_data = normalizer.normalize_data(raw_data, platform)
        
        # 데이터프레임으로 변환
        df = pd.DataFrame(normalized_data)
        
        # 데이터 검증
        validator = DataValidator()
        validation_results = validator.validate_data(df)
        
        # 검증 결과 로깅
        if any(validation_results.values()):
            logger.warning(f"데이터 검증 결과: {validation_results}")
        
        return df
        
    except Exception as e:
        logger.error(f"데이터 로드 및 정규화 중 오류 발생: {str(e)}")
        raise 