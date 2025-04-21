from abc import ABC, abstractmethod
import logging
from datetime import datetime
import os
import re
import emoji
import requests

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