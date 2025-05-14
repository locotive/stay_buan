from abc import ABC, abstractmethod
import logging
from datetime import datetime
import os
import re
import emoji
import requests
import time

class BaseCrawler(ABC):
    """모든 크롤러의 기본 클래스"""
    
    def __init__(self, keywords, max_pages=1, save_dir="data/raw"):
        self.keywords = keywords if isinstance(keywords, list) else [keywords]
        self.max_pages = max_pages
        self.save_dir = save_dir
        self.platform = self.__class__.__name__.replace("Crawler", "").lower()
        
        # 저장 디렉토리 생성
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.platform)
        
    @abstractmethod
    def crawl(self):
        """각 플랫폼별 크롤링 로직 구현"""
        pass
    
    def clean_text(self, text):
        """텍스트 정제 (이모지 제거, 공백 정리 등)"""
        if not text:
            return ""
        
        # 이모지 제거
        text = emoji.replace_emoji(text, "")
        
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        
        # 불필요한 공백 및 특수문자 정리
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def generate_filename(self, keyword):
        """저장 파일명 생성"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_keyword = re.sub(r'[^\w\s]', '', keyword).replace(' ', '_')
        return f"{self.platform}_{safe_keyword}_{timestamp}"

    def is_crawling_allowed(self, url):
        """robots.txt 무시 (우회 허용)"""
        self.logger.warning(f"Bypassing robots.txt check for {url}")
        return True  # 항상 허용으로 처리

    def _normalize_date(self, date_str):
        """날짜를 YYYYMMDD 형식의 문자열로 변환"""
        if not date_str:
            return time.strftime("%Y%m%d")
        
        try:
            # 문자열로 변환
            date_str = str(date_str).strip()
            
            # 이미 YYYYMMDD 형식인 경우
            if date_str.isdigit() and len(date_str) == 8:
                return date_str
            
            # 날짜 형식 변환
            date_formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y.%m.%d %H:%M:%S",
                "%Y-%m-%d",
                "%Y.%m.%d",
                "%Y%m%d",
                "%Y년%m월%d일",
                "%Y년 %m월 %d일"
            ]
            
            # 각 형식 시도
            for fmt in date_formats:
                try:
                    date_obj = datetime.strptime(date_str, fmt)
                    return date_obj.strftime("%Y%m%d")
                except:
                    continue
                
            # 숫자만 추출 시도 (20230101 형식)
            digits = ''.join(filter(str.isdigit, date_str))
            if len(digits) >= 8:
                return digits[:8]
            
            # 모든 시도 실패 시 현재 날짜 반환
            return time.strftime("%Y%m%d")
        
        except Exception as e:
            self.logger.warning(f"날짜 변환 실패: {date_str} - {str(e)}")
            return time.strftime("%Y%m%d")

    def validate_platform_data(self, df, platform):
        """플랫폼별 데이터 유효성 검사"""
        validation_results = {
            'missing_fields': [],
            'invalid_dates': [],
            'invalid_urls': [],
            'empty_content': []
        }
        
        # 1. 필수 필드 확인
        required_fields = self.get_required_fields(platform)
        for field in required_fields:
            if field not in df.columns:
                validation_results['missing_fields'].append(field)
        
        # 2. 날짜 형식 검사
        if 'published_date' in df.columns:
            # 숫자형인 경우 문자열로 변환
            df['published_date'] = df['published_date'].astype(str)
            # 8자리 숫자 형식이 아닌 날짜 찾기
            invalid_dates = df[~df['published_date'].str.match(r'^\d{8}$')]
            if not invalid_dates.empty:
                # 잘못된 날짜 형식 수정
                df.loc[invalid_dates.index, 'published_date'] = df.loc[invalid_dates.index, 'published_date'].apply(
                    lambda x: self._normalize_date(x)
                )
                validation_results['invalid_dates'] = invalid_dates['published_date'].tolist()
        
        # 3. URL 형식 검사
        if 'url' in df.columns:
            invalid_urls = df[~df['url'].str.startswith(('http://', 'https://'))]
            if not invalid_urls.empty:
                validation_results['invalid_urls'] = invalid_urls['url'].tolist()
        
        # 4. 빈 컨텐츠 확인
        if 'content' in df.columns:
            empty_content = df[df['content'].isna() | (df['content'].str.strip() == '')]
            if not empty_content.empty:
                validation_results['empty_content'] = empty_content.index.tolist()
        
        return validation_results