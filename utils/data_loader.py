import os
import json
from datetime import datetime

class DataLoader:
    """데이터 로딩 및 필터링 유틸리티"""
    
    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def load_data(self, platform=None, keyword=None, start_date=None, end_date=None):
        """JSON 데이터 로드 및 필터링"""
        all_data = []
        
        # 데이터 디렉토리 내의 모든 파일 탐색
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.data_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_data.extend(data)
        
        # 필터링
        filtered_data = self.filter_data(all_data, platform, keyword, start_date, end_date)
        return filtered_data
    
    def filter_data(self, data, platform, keyword, start_date, end_date):
        """데이터 필터링"""
        filtered_data = []
        
        for item in data:
            # 플랫폼 필터
            if platform and item.get('platform') != platform:
                continue
            
            # 키워드 필터
            if keyword and keyword not in item.get('keyword', ''):
                continue
            
            # 날짜 필터
            item_date = datetime.strptime(item.get('crawled_at', ''), "%Y-%m-%d %H:%M:%S")
            if start_date and item_date < start_date:
                continue
            if end_date and item_date > end_date:
                continue
            
            # 필드 통합
            unified_item = {
                'title': item.get('title', ''),
                'content': item.get('content', ''),
                'date': item.get('crawled_at', ''),
                'sentiment': item.get('sentiment', None),
                'platform': item.get('platform', ''),
                'keyword': item.get('keyword', ''),
                'lat': item.get('lat', None),
                'lon': item.get('lon', None)
            }

            filtered_data.append(unified_item)
        
        return filtered_data