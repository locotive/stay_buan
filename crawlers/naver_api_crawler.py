import time
import os
import json
import random
import hashlib
from urllib.parse import quote
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from datetime import datetime
from utils.cache import JsonCache
from core.base_crawler import BaseCrawler

load_dotenv(override=True)

class DuplicateDocError(Exception):
    """문서 중복 오류"""
    pass

def contains_all(text, keyword_list):
    """텍스트에 모든 키워드가 포함되어 있는지 확인"""
    if not text or not keyword_list:
        return False
    
    text = text.lower()
    for kw in keyword_list:
        if kw.lower() not in text:
            return False
    return True

class NaverBlogCrawler:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def get_blog_content(self, url):
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                if 'blog.naver.com' in url:
                    iframe = soup.find('iframe', id='mainFrame')
                    if iframe:
                        iframe_url = 'https://blog.naver.com' + iframe.get('src')
                        response = requests.get(iframe_url, headers=self.headers)
                        soup = BeautifulSoup(response.text, 'html.parser')
                    content = soup.find('div', class_='se-main-container')
                    if content:
                        return content.get_text(strip=True)
                elif 'news.naver.com' in url:
                    content = soup.find('div', id='dic_area')
                    if content:
                        return content.get_text(strip=True)
            return None
        except Exception as e:
            print(f"Error crawling content from {url}: {str(e)}")
            return None

class NaverSearchAPICrawler(BaseCrawler):
    def __init__(self, keywords, max_pages=1000, save_dir="data/raw"):
        super().__init__(keywords, max_pages=max_pages, save_dir=save_dir)
        self.targets = ["blog", "news"]

        self.client_id = os.getenv("NAVER_CLIENT_ID")
        self.client_secret = os.getenv("NAVER_CLIENT_SECRET")

        if not self.client_id or not self.client_secret:
            self.logger.error("네이버 API 키가 없습니다.")
            raise ValueError("네이버 API 키가 없습니다.")

        self.headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret
        }
        self.cache = JsonCache()
        self.blog_crawler = NaverBlogCrawler()
        self.doc_ids = set()  # 문서 ID 중복 체크를 위한 세트

        # [핵심] keywords를 문자열 리스트로 변환
        self.keywords = [k['text'] for k in keywords]

    def _postprocess(self, items, keywords):
        """
        수집된 아이템을 후처리하는 메서드
        
        1. 모든 키워드가 제목+내용에 포함된 경우만 선택
        2. URL+제목 기반 해시로 중복 제거
        """
        processed_items = []
        
        for item in items:
            # 1. 모든 키워드 포함 확인
            combined_text = f"{item['title']} {item['content']}"
            if not contains_all(combined_text, keywords):
                self.logger.info(f"키워드 필터링: 모든 키워드를 포함하지 않음 - {item['title']}")
                continue
                
            # 2. 중복 문서 확인
            doc_id = hashlib.sha256((item['url'] + item['title']).encode()).hexdigest()
            if doc_id in self.doc_ids:
                self.logger.info(f"중복 문서: {item['title']}")
                raise DuplicateDocError(f"중복 문서 ID: {doc_id}")
            
            # 문서 ID 추가
            self.doc_ids.add(doc_id)
            item['doc_id'] = doc_id
            
            processed_items.append(item)
            
        return processed_items

    def generate_keyword_variations(self, keywords):
        if not keywords:
            return []
        region_keyword = keywords[0]
        additional_keywords = keywords[1:]

        variations = []
        if region_keyword:
            and_combination = region_keyword
            if additional_keywords:
                and_combination += " " + " ".join(additional_keywords)
            variations.append(and_combination)

        variations.extend(additional_keywords)
        return list(set(variations))

    def crawl(self):
        combined_results = []
        try:
            for target in self.targets:
                self.logger.info(f"네이버 {target} API 기반 수집 시작")
                api_url = f"https://openapi.naver.com/v1/search/{target}.json"

                keyword_variations = self.generate_keyword_variations(self.keywords)
                self.logger.info(f"생성된 키워드 변형: {keyword_variations}")

                for keyword in keyword_variations:
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
                            url = item.get("link", "")
                            blog_name = item.get("bloggername", item.get("author", ""))
                            date = item.get("postdate", item.get("pubDate", "")).replace("-", "")[:8]

                            if url in seen_urls:
                                continue
                            seen_urls.add(url)

                            full_content = self.blog_crawler.get_blog_content(url)
                            content = full_content if full_content else f"{title} {desc}"

                            try:
                                date_obj = datetime.strptime(date, "%Y%m%d")
                            except:
                                date_obj = datetime.now()

                            post_data = {
                                "title": title,
                                "content": content,
                                "url": url,
                                "blog_name": blog_name,
                                "published_date": date,
                                "date_obj": date_obj.isoformat(),
                                "platform": f"naver_{target}_api",
                                "keyword": keyword,
                                "original_keywords": ",".join(self.keywords),
                                "sentiment": None,
                                "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            }
                            keyword_results.append(post_data)
                            self.cache.save(url)

                        page += 1
                        time.sleep(random.uniform(0.3, 1))

                    # 후처리 적용
                    try:
                        keyword_results = self._postprocess(keyword_results, self.keywords)
                    except DuplicateDocError as e:
                        self.logger.warning(str(e))
                    
                    keyword_results.sort(key=lambda x: x['date_obj'], reverse=True)

                    self.logger.info(f"[SUMMARY] Total {len(keyword_results)} items collected for '{keyword}'")
                    if keyword_results:
                        keywords_str = '_'.join(self.keywords)
                        filename = f"naver_blog_{len(keyword_results)}_{keywords_str}_{time.strftime('%Y%m%d_%H%M%S')}.json"
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
                unique_results = {item['url']: item for item in combined_results}.values()
                combined_results = list(unique_results)
                combined_results.sort(key=lambda x: x['date_obj'], reverse=True)

                keywords_str = '_'.join(self.keywords)
                filename = f"naver_blog_{len(combined_results)}_{keywords_str}_combined_{time.strftime('%Y%m%d_%H%M%S')}.json"
                filepath = os.path.join(self.save_dir, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(combined_results, f, ensure_ascii=False, indent=2)
                self.logger.info(f"Saved combined results to {filepath}")
            else:
                self.logger.warning("No data saved (empty result set)")

        except Exception as e:
            self.logger.error(f"Crawler error: {str(e)}")

        return combined_results