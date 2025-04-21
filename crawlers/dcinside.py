import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import quote

class DCInsideCrawler(BaseCrawler):
    """디시인사이드 크롤러"""
    
    def __init__(self, keywords, max_pages=3, save_dir="data/raw"):
        super().__init__(keywords, max_pages, save_dir)
        self.base_url = "https://gall.dcinside.com/board/lists/?id="
    
    def crawl(self):
        """디시인사이드 데이터 수집"""
        all_results = []
        
        for keyword in self.keywords:
            self.logger.info(f"Crawling DCInside for keyword: {keyword}")
            encoded_keyword = quote(keyword)
            keyword_results = []
            
            for page in range(1, self.max_pages + 1):
                response = requests.get(f"{self.base_url}{encoded_keyword}&page={page}")
                soup = BeautifulSoup(response.text, 'html.parser')
                
                posts = soup.select('tr.ub-content')
                for post in posts:
                    title_tag = post.select_one('a.subject')
                    if title_tag:
                        post_data = {
                            'title': title_tag.get_text(strip=True),
                            'content': "Example content",  # 실제로는 게시글 본문을 수집해야 함
                            'url': title_tag['href'],
                            'platform': 'dcinside',
                            'keyword': keyword,
                            'published_date': "2025-04-07 20:33:36",  # 실제로는 게시글 날짜를 수집해야 함
                            'crawled_at': time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        keyword_results.append(post_data)
                
                time.sleep(1)  # 페이지 간 딜레이
            
            # keyword별로 저장
            if keyword_results:
                filename = self.generate_filename(keyword)
                DataSaver.save_json(keyword_results, filename, self.save_dir)
                self.logger.info(f"Saved {len(keyword_results)} results for keyword '{keyword}'")
        
        return all_results 