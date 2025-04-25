import time
import os
import json
import random
from urllib.parse import quote
import requests
from dotenv import load_dotenv
from datetime import datetime
from utils.cache import JsonCache
from core.base_crawler import BaseCrawler
from itertools import combinations

load_dotenv()

class NaverSearchAPICrawler(BaseCrawler):
    def __init__(self, keywords, max_pages=3, save_dir="data/raw"):
        super().__init__(keywords, max_pages=max_pages, save_dir=save_dir)
        self.targets = ["blog", "news"]  # cafearticle 제외
        self.client_id = os.getenv("NAVER_CLIENT_ID")
        self.client_secret = os.getenv("NAVER_CLIENT_SECRET")
        self.headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret
        }
        self.cache = JsonCache()
        
    def generate_keyword_variations(self, keyword):
        """키워드 변형 생성"""
        variations = [keyword]  # 원본 키워드
        
        # 키워드에 공백이 있는 경우 분리
        if ' ' in keyword:
            words = keyword.split()
            variations.extend(words)  # 개별 단어 추가
            
            # 2개 이상의 단어가 있는 경우 조합 생성
            if len(words) >= 2:
                for r in range(2, min(len(words) + 1, 4)):  # 최대 3개 단어 조합까지
                    for combo in combinations(words, r):
                        variations.append(' '.join(combo))
        
        return list(set(variations))  # 중복 제거

    def crawl(self):
        combined_results = []
        try:
            for target in self.targets:
                self.logger.info(f"네이버 {target} API 기반 수집 시작")
                api_url = f"https://openapi.naver.com/v1/search/{target}.json"

                # 각 키워드에 대한 변형 생성
                all_keywords = []
                for keyword in self.keywords:
                    variations = self.generate_keyword_variations(keyword)
                    all_keywords.extend(variations)
                all_keywords = list(set(all_keywords))  # 중복 제거
                
                self.logger.info(f"생성된 키워드 변형: {all_keywords}")

                for keyword in all_keywords:
                    self.logger.info(f"Crawling for keyword: {keyword}")
                    encoded_keyword = quote(keyword)
                    keyword_results = []
                    seen_urls = set()

                    page = 0
                    while True:
                        if self.max_pages and page >= self.max_pages:
                            break

                        start = page * 10 + 1
                        if start >= 1000:
                            self.logger.warning(f"Reached API's max limit (1000 items) for keyword: {keyword}")
                            break

                        params = {
                            "query": encoded_keyword,
                            "display": 10,
                            "start": start,
                            "sort": "date"
                        }

                        res = requests.get(api_url, headers=self.headers, params=params)
                        if res.status_code != 200:
                            self.logger.error(f"API Error [{res.status_code}]: {res.text}")
                            break

                        items = res.json().get("items", [])
                        self.logger.info(f"Found {len(items)} items on page {page + 1} for keyword: {keyword}")
                        if not items:
                            break

                        for item in items:
                            title = self.clean_text(item.get("title", ""))
                            desc = self.clean_text(item.get("description", ""))
                            content = f"{title} {desc}"
                            url = item.get("link", "")
                            blog_name = item.get("bloggername", item.get("author", ""))
                            date = item.get("postdate", item.get("pubDate", "")).replace("-", "")[:8]

                            # 중복 필터
                            if self.cache.exists(url):
                                continue

                            post_data = {
                                "title": title,
                                "content": content,
                                "url": url,
                                "blog_name": blog_name,
                                "published_date": date,
                                "platform": f"naver_{target}_api",
                                "keyword": keyword,
                                "original_keywords": ",".join(self.keywords),
                                "sentiment": None,
                                "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            }
                            keyword_results.append(post_data)
                            self.cache.save(url)

                        page += 1
                        time.sleep(random.uniform(0.7, 1.5))

                    self.logger.info(f"[SUMMARY] Total {len(keyword_results)} items collected for '{keyword}'")
                    if keyword_results:
                        filename = f"{self.platform}_{target}_{keyword}_{time.strftime('%Y%m%d_%H%M%S')}.json"
                        filepath = os.path.join(self.save_dir, filename)
                        os.makedirs(self.save_dir, exist_ok=True)
                        with open(filepath, "w", encoding="utf-8") as f:
                            json.dump(keyword_results, f, ensure_ascii=False, indent=2)
                        self.logger.info(f"Saved results to {filepath}")
                        combined_results.extend(keyword_results)
                    else:
                        self.logger.warning(f"No data saved for keyword '{keyword}'")

            self.logger.info(f"[FINAL SUMMARY] Total {len(combined_results)} items collected for all keywords")
            if combined_results:
                # 중복 제거
                unique_results = {item['url']: item for item in combined_results}.values()
                combined_results = list(unique_results)
                
                # 최종 결과 저장
                filename = f"{self.platform}_combined_{time.strftime('%Y%m%d_%H%M%S')}.json"
                filepath = os.path.join(self.save_dir, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(combined_results, f, ensure_ascii=False, indent=2)
                self.logger.info(f"Saved combined results to {filepath}")
            else:
                self.logger.warning("No data saved (empty result set)")

        except Exception as e:
            self.logger.error(f"Crawler error: {str(e)}")

        return combined_results
