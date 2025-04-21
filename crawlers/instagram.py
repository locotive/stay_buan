from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import quote
import time
import random

from core.base_crawler import BaseCrawler
from utils.savers import DataSaver

class InstagramCrawler(BaseCrawler):
    """인스타그램 해시태그 기반 크롤러"""

    def __init__(self, keywords, max_pages=3, save_dir="data/raw"):
        super().__init__(keywords, max_pages, save_dir)
        self.base_url = "https://www.instagram.com/explore/tags/"

    def setup_driver(self):
        """ChromeDriver 설정 및 초기화"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver

    def crawl(self):
        self.logger.info("robots.txt 검사 우회 중 (Instagram 강제 실행)")

        all_results = []
        driver = self.setup_driver()

        try:
            for keyword in self.keywords:
                self.logger.info(f"Crawling Instagram for keyword: {keyword}")
                encoded_keyword = quote(keyword)
                keyword_results = []
                
                driver.get(f"https://www.instagram.com/explore/tags/{encoded_keyword}/")
                time.sleep(3)

                for _ in range(self.max_pages):
                    posts = driver.find_elements(By.CSS_SELECTOR, "article div div div div a")
                    for post in posts:
                        url = post.get_attribute("href")
                        if url:
                            item = {
                                'title': f'Instagram post for #{keyword}',
                                'content': f'Instagram link for #{keyword}: {url}',
                                'url': url,
                                'platform': 'instagram',
                                'keyword': keyword,
                                'crawled_at': time.strftime("%Y-%m-%d %H:%M:%S")
                            }
                            keyword_results.append(item)

                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(random.uniform(2, 4))

                if keyword_results:
                    filename = self.generate_filename(keyword)
                    DataSaver.save_json(keyword_results, filename, self.save_dir)
                    self.logger.info(f"Saved {len(keyword_results)} results for keyword '{keyword}'")

        finally:
            driver.quit()

        return all_results
