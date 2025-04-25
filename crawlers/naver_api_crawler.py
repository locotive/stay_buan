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

load_dotenv()

class NaverSearchAPICrawler(BaseCrawler):
    def __init__(self, keywords, start_date=None, end_date=None, save_dir="data/raw",
                 start_page=1, end_page=50):
        super().__init__(keywords, max_pages=None, save_dir=save_dir)
        self.targets = ["blog", "news"]  # cafearticle 제외
        self.client_id = os.getenv("NAVER_CLIENT_ID")
        self.client_secret = os.getenv("NAVER_CLIENT_SECRET")
        self.headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret
        }
        self.start_date = start_date
        self.end_date = end_date
        self.start_page = start_page
        self.end_page = end_page
        self.cache = JsonCache()

    def crawl(self):
        combined_results = []
        try:
            for target in self.targets:
                self.logger.info(f"네이버 {target} API 기반 수집 시작")
                api_url = f"https://openapi.naver.com/v1/search/{target}.json"

                for page in range(self.start_page, self.end_page + 1):
                    start = (page - 1) * 10 + 1
                    if start >= 1000:
                        self.logger.warning("Reached API's max limit (1000 items per keyword/target).")
                        break

                    query = " ".join(self.keywords)
                    params = {
                        "query": quote(query),
                        "display": 10,
                        "start": start,
                        "sort": "date"
                    }

                    res = requests.get(api_url, headers=self.headers, params=params)
                    if res.status_code != 200:
                        self.logger.error(f"API Error [{res.status_code}]: {res.text}")
                        break

                    items = res.json().get("items", [])
                    self.logger.info(f"Found {len(items)} items on page {page}")

                    for item in items:
                        title = self.clean_text(item.get("title", ""))
                        desc = self.clean_text(item.get("description", ""))
                        content = f"{title} {desc}"
                        url = item.get("link", "")
                        blog_name = item.get("bloggername", item.get("author", ""))
                        raw_date = item.get("postdate", item.get("pubDate", ""))
                        date = raw_date.replace("-", "")[:8]

                        # ✅ AND 키워드 필터, 날짜 필터, 중복 필터
                        if not self._is_date_in_range(date):
                            continue
                        if self.cache.exists(url):
                            continue
                        if not all(k in content for k in self.keywords):
                            continue

                        post_data = {
                            "title": title,
                            "content": content,
                            "url": url,
                            "blog_name": blog_name,
                            "published_date": date,
                            "platform": f"naver_{target}_api",
                            "keyword": ",".join(self.keywords),
                            "sentiment": None,
                            "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        combined_results.insert(0, post_data)
                        self.cache.save(url)

                    time.sleep(random.uniform(0.7, 1.5))

            self.logger.info(f"[SUMMARY] Total {len(combined_results)} items collected for keywords: {self.keywords}")
            if combined_results:
                from_date = combined_results[-1]["published_date"]
                to_date = combined_results[0]["published_date"]
                keywords_joined = "_".join(self.keywords)
                filename = f"{self.platform}_{keywords_joined}_{self.start_page}p_{self.end_page}p.json"
                filepath = os.path.join(self.save_dir, filename)
                os.makedirs(self.save_dir, exist_ok=True)
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(combined_results, f, ensure_ascii=False, indent=2)
                self.logger.info(f"Saved results to {filepath}")
            else:
                self.logger.warning("No data saved (empty result set)")

        except Exception as e:
            self.logger.error(f"Crawler error: {str(e)}")

        return combined_results

    def _is_date_in_range(self, date_str):
        if not self.start_date or not self.end_date:
            return True
        return self.start_date <= date_str <= self.end_date
