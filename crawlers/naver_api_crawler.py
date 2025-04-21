import time
import os
import json
import random
from urllib.parse import quote
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta

from core.base_crawler import BaseCrawler
from utils.savers import DataSaver

load_dotenv()


class NaverSearchAPICrawler(BaseCrawler):
    """네이버 통합검색 API 기반 크롤러 (뉴스/블로그/카페 등 확장 가능)"""

    def __init__(self, keywords, start_date=None, end_date=None, save_dir="data/raw", targets=None):
        super().__init__(keywords, max_pages=None, save_dir=save_dir)
        self.targets = targets if targets else ["blog", "news", "cafearticle"]
        self.client_id = os.getenv("NAVER_CLIENT_ID")
        self.client_secret = os.getenv("NAVER_CLIENT_SECRET")
        self.headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret
        }
        self.start_date = start_date
        self.end_date = end_date

    def crawl(self):
        all_results = []
        try:
            for target in self.targets:
                self.logger.info(f"네이버 {target} API 기반 수집 시작")
                api_url = f"https://openapi.naver.com/v1/search/{target}.json"

                for keyword in self.keywords:
                    self.logger.info(f"Crawling for keyword: {keyword}")
                    encoded_keyword = quote(keyword)
                    keyword_results = []
                    seen_urls = set()

                    start_dt = datetime.strptime(self.start_date, "%Y%m%d")
                    end_dt = datetime.strptime(self.end_date, "%Y%m%d")
                    delta = timedelta(days=1)

                    while start_dt <= end_dt:
                        date_str = start_dt.strftime("%Y%m%d")
                        self.logger.info(f"  📆 {date_str} 수집 중")

                        page = 0
                        while True:
                            start = page * 10 + 1
                            if start >= 1000:
                                self.logger.warning("Reached API's max limit (1000 items per keyword/target/date).")
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
                            self.logger.info(f"Found {len(items)} items on page {page + 1} for {date_str}")
                            if not items:
                                break

                            found_any = False
                            for item in items:
                                title = self.clean_text(item.get("title", ""))
                                desc = self.clean_text(item.get("description", ""))
                                content = f"{title} {desc}"
                                url = item.get("link")
                                blog_name = item.get("bloggername", item.get("author", ""))
                                date = item.get("postdate", item.get("pubDate", ""))

                                if not date or url in seen_urls:
                                    continue

                                # 날짜 형식이 YYYYMMDD 또는 YYYY-MM-DD 형태일 수 있음
                                cleaned_date = date.replace("-", "")[:8]
                                if cleaned_date != date_str:
                                    continue

                                if content and len(content) >= 10 and url:
                                    seen_urls.add(url)
                                    post_data = {
                                        "title": title,
                                        "content": content,
                                        "url": url,
                                        "blog_name": blog_name,
                                        "published_date": cleaned_date,
                                        "platform": f"naver_{target}_api",
                                        "keyword": keyword,
                                        "sentiment": None,
                                        "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    }
                                    keyword_results.insert(0, post_data)  # 최신순으로 삽입
                                    found_any = True

                            if not found_any:
                                break

                            page += 1
                            time.sleep(random.uniform(0.3, 0.7))

                        start_dt += delta

                    self.logger.info(f"[SUMMARY] Total {len(keyword_results)} items collected for '{keyword}'")
                    if keyword_results:
                        from_date = keyword_results[-1]["published_date"]
                        to_date = keyword_results[0]["published_date"]
                        filename = f"{self.platform}_{target}_{keyword}_{from_date}_{to_date}.json"
                        filepath = os.path.join(self.save_dir, filename)
                        os.makedirs(self.save_dir, exist_ok=True)
                        with open(filepath, "w", encoding="utf-8") as f:
                            json.dump(keyword_results, f, ensure_ascii=False, indent=2)
                        self.logger.info(f"Saved {len(keyword_results)} results for keyword '{keyword}' → {filepath}")
                        all_results.extend(keyword_results)
                    else:
                        self.logger.warning(f"No data saved for keyword '{keyword}'")

        except Exception as e:
            self.logger.error(f"Crawler error: {str(e)}")

        return all_results
