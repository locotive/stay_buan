import re
import json
from bs4 import BeautifulSoup

class HTMLParser:
    """HTML 파싱 유틸리티"""
    
    @staticmethod
    def extract_text(element):
        """BS4 엘리먼트에서 텍스트 추출"""
        if not element:
            return ""
        return element.get_text(strip=True)
    
    @staticmethod
    def extract_attribute(element, attr):
        """BS4 엘리먼트에서 속성 추출"""
        if not element:
            return ""
        return element.get(attr, "")
    
    @staticmethod
    def extract_date(text):
        """텍스트에서 날짜 추출"""
        # 날짜 패턴 예: 2023-10-15, 2023.10.15, 2023/10/15
        date_pattern = r'(\d{4}[-./]\d{1,2}[-./]\d{1,2})'
        match = re.search(date_pattern, text)
        if match:
            return match.group(1)
        return "" 