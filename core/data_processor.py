import re
import pandas as pd
from collections import Counter

class DataProcessor:
    """수집된 데이터 처리 클래스"""
    
    @staticmethod
    def remove_duplicates(data_list, key_field='url'):
        """URL 기준 중복 제거"""
        seen = set()
        unique_data = []
        
        for item in data_list:
            identifier = item.get(key_field, '')
            if identifier and identifier not in seen:
                seen.add(identifier)
                unique_data.append(item)
                
        return unique_data
    
    @staticmethod
    def filter_spam(data_list, content_field='content', threshold=0.7):
        """스팸 데이터 필터링 (단순 구현)"""
        filtered_data = []
        
        for item in data_list:
            content = item.get(content_field, '')
            
            # 광고 키워드 체크
            ad_keywords = ['광고', '홍보', '구매', '할인', '이벤트', '쿠폰', '공동구매']
            ad_count = sum(1 for kw in ad_keywords if kw in content)
            
            # URL 과다 체크
            url_count = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content))
            
            # 짧은 컨텐츠 체크
            if len(content) < 20 or ad_count > 3 or url_count > 3:
                continue
                
            filtered_data.append(item)
                
        return filtered_data
    
    @staticmethod
    def extract_keywords(data_list, content_field='content'):
        """데이터에서 주요 키워드 추출"""
        all_text = ' '.join([item.get(content_field, '') for item in data_list])
        
        # 불용어 설정
        stopwords = {'있는', '없는', '그리고', '그러나', '그런데', '때문에', '이런', '저런', '어떤', '그것'}
        
        # 단어 추출 및 카운트
        words = re.findall(r'\w+', all_text)
        words = [w for w in words if len(w) > 1 and w not in stopwords]
        
        return Counter(words).most_common(20) 