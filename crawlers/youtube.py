import requests
import time
import random
from urllib.parse import quote

from core.base_crawler import BaseCrawler
from utils.savers import DataSaver

class YouTubeCrawler(BaseCrawler):
    """유튜브 크롤러"""

    def __init__(self, keywords, max_results=30, save_dir="data/raw"):
        super().__init__(keywords, max_results, save_dir)
        self.api_key = "YOUR_YOUTUBE_API_KEY"
        self.base_url = "https://www.googleapis.com/youtube/v3"

    def crawl(self):
        """유튜브 데이터 수집"""
        all_results = []
        
        for keyword in self.keywords:
            self.logger.info(f"Crawling YouTube for keyword: {keyword}")
            encoded_keyword = quote(keyword)
            keyword_results = []
            
            search_url = f"{self.base_url}/search"
            params = {
                'part': 'snippet',
                'q': encoded_keyword,
                'type': 'video',
                'maxResults': self.max_results,
                'key': self.api_key
            }
            
            response = requests.get(search_url, params=params)
            if response.status_code == 200:
                videos = response.json().get('items', [])
                for video in videos:
                    video_data = {
                        'title': video['snippet']['title'],
                        'content': video['snippet']['description'],
                        'url': f"https://www.youtube.com/watch?v={video['id']['videoId']}",
                        'platform': 'youtube',
                        'keyword': keyword,
                        'published_date': video['snippet']['publishedAt'],
                        'crawled_at': time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    keyword_results.append(video_data)
            
            # keyword별로 저장
            if keyword_results:
                filename = self.generate_filename(keyword)
                DataSaver.save_json(keyword_results, filename, self.save_dir)
                self.logger.info(f"Saved {len(keyword_results)} results for keyword '{keyword}'")
        
        return all_results
