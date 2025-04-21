import requests
from bs4 import BeautifulSoup
import random
import time
from urllib.parse import quote

from core.base_crawler import BaseCrawler
from utils.savers import DataSaver

class DaumNewsCrawler(BaseCrawler):
    """다음 뉴스 크롤러"""

    def __init__(self, keywords, max_pages=3, save_dir="data/raw"):
        super().__init__(keywords, max_pages, save_dir)
        self.base_url = "https://search.daum.net/search?w=news&q="

    def crawl(self):
        """다음 뉴스 데이터 수집"""
        all_results = []
        
        for keyword in self.keywords:
            self.logger.info(f"Crawling Daum News for keyword: {keyword}")
            encoded_keyword = quote(keyword)
            keyword_results = []
            
            for page in range(1, self.max_pages + 1):
                response = requests.get(f"{self.base_url}{encoded_keyword}&p={page}")
                soup = BeautifulSoup(response.text, 'html.parser')
                
                articles = soup.select('a.f_link_b')
                for article in articles:
                    article_data = {
                        'title': article.get_text(),
                        'content': "Example content",  # 실제로는 기사 본문을 수집해야 함
                        'url': article['href'],
                        'platform': 'daum_news',
                        'keyword': keyword,
                        'published_date': "2025-04-07 20:33:36",  # 실제로는 기사 날짜를 수집해야 함
                        'crawled_at': time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    keyword_results.append(article_data)
                
                time.sleep(1)  # 페이지 간 딜레이
            
            # keyword별로 저장
            if keyword_results:
                filename = self.generate_filename(keyword)
                DataSaver.save_json(keyword_results, filename, self.save_dir)
                self.logger.info(f"Saved {len(keyword_results)} results for keyword '{keyword}'")
        
        return all_results
