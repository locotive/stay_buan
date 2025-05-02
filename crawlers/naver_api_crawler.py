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
from utils.content_extractor import extract_content
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import sys
import itertools
import logging
import re

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

def contains_any(text, keyword_list):
    """텍스트에 키워드 중 하나라도 포함되어 있는지 확인"""
    if not text or not keyword_list:
        return False
    
    text = text.lower()
    for kw in keyword_list:
        if kw.lower() in text:
            return True
    return False

def check_keyword_conditions(text, keywords):
    """키워드 AND/OR 조건을 검사"""
    if not text or not keywords:
        return False
    
    # 지역 키워드 (항상 포함되어야 함)
    region_keyword = keywords[0]['text'] if keywords else None
    if not region_keyword or region_keyword.lower() not in text.lower():
        return False
    
    # 키워드가 1개만 있으면 (지역 키워드만) 무조건 통과
    if len(keywords) == 1:
        return True
    
    # AND 키워드가 있는지 확인
    and_keywords = [k['text'] for k in keywords[1:] if k['condition'] == 'AND' and k['text']]
    # OR 키워드가 있는지 확인
    or_keywords = [k['text'] for k in keywords[1:] if k['condition'] == 'OR' and k['text']]
    
    # AND 키워드가 없고 OR 키워드도 없으면, 지역 키워드만 있으므로 통과
    if not and_keywords and not or_keywords:
        return True
    
    # AND 키워드가 있고 OR 키워드가 없는 경우: 지역키워드 + AND 키워드 중 하나라도 포함되면 통과 (완화된 조건)
    if and_keywords and not or_keywords:
        for kw in and_keywords:
            if kw.lower() in text.lower():
                return True
        return False  # 어떤 AND 키워드도 포함되지 않음
    
    # AND 키워드가 없고 OR 키워드만 있는 경우: 지역키워드 + OR 키워드 중 하나라도 포함되면 통과
    if not and_keywords and or_keywords:
        for kw in or_keywords:
            if kw.lower() in text.lower():
                return True
        return False  # 어떤 OR 키워드도 포함되지 않음
    
    # AND 키워드와 OR 키워드가 모두 있는 경우
    # AND 키워드 중 하나라도 포함되고 OR 키워드 중 하나라도 포함되면 통과 (완화된 조건)
    and_found = False
    for kw in and_keywords:
        if kw.lower() in text.lower():
            and_found = True
            break
    
    or_found = False
    for kw in or_keywords:
        if kw.lower() in text.lower():
            or_found = True
            break
    
    return and_found and or_found

class NaverBlogCrawler:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Selenium 웹드라이버 설정
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.driver = None
        self.init_attempts = 0  # 드라이버 초기화 시도 횟수
        
    def init_driver(self):
        """필요시 드라이버 초기화"""
        if self.driver is None and self.init_attempts < 2:  # 최대 2번만 시도
            self.init_attempts += 1
            try:
                # 먼저 크롬 버전을 확인하지 않도록 설정
                os.environ['WDM_LOG_LEVEL'] = '0'  # 로그 레벨 낮춤
                os.environ['WDM_PRINT_FIRST_LINE'] = 'False'  # 첫 줄 출력 안함
                
                # Chrome이 설치되어 있지 않을 경우 대체 옵션
                try:
                    # 직접 chromium 패스 설정 (macOS)
                    if sys.platform == "darwin":  # macOS
                        chrome_path = "/Applications/Chromium.app/Contents/MacOS/Chromium"
                        if os.path.exists(chrome_path):
                            self.chrome_options.binary_location = chrome_path
                    
                    # 드라이버 초기화 시도
                    self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.chrome_options)
                except Exception as e1:
                    # 실패 시 Firefox 사용 시도
                    try:
                        from selenium.webdriver.firefox.service import Service as FirefoxService
                        from webdriver_manager.firefox import GeckoDriverManager
                        
                        firefox_options = webdriver.FirefoxOptions()
                        firefox_options.add_argument("--headless")
                        self.driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=firefox_options)
                    except Exception as e2:
                        # 동적 크롤링 비활성화
                        print(f"드라이버 초기화 실패: {str(e1)}\n두 번째 시도 실패: {str(e2)}")
                        print("동적 크롤링이 비활성화됩니다. 정적 크롤링만 사용합니다.")
                        self.driver = None
            except Exception as e:
                print(f"드라이버 초기화 실패: {str(e)}")
                self.driver = None
                
    def close_driver(self):
        """드라이버 종료"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None

    def get_blog_content(self, url):
        """
        블로그 콘텐츠 추출을 위한 메서드
        content_extractor 모듈을 사용하여 URL에서 콘텐츠 추출
        """
        if not url:
            return None
            
        # content_extractor 모듈의 extract_content 함수 사용
        content = extract_content(url, self.driver)
        
        # 콘텐츠가 없고 드라이버가 초기화되지 않은 경우
        if not content and self.driver is None:
            self.init_driver()
            if self.driver:
                # 드라이버 초기화 성공 시 다시 시도
                content = extract_content(url, self.driver)
                
        # 콘텐츠 추출 결과 로깅
        if content:
            return self._clean_content(content)
        
        return None
        
    def _clean_content(self, content):
        """추출된 콘텐츠 정제"""
        if not content:
            return None
            
        # 중복 공백 제거
        content = re.sub(r'\s+', ' ', content)
        
        # HTML 태그 제거 (혹시 남아있는 경우)
        content = re.sub(r'<[^>]*>', '', content)
        
        # 특수 문자 정리
        content = content.replace('&nbsp;', ' ')
        content = content.replace('&lt;', '<')
        content = content.replace('&gt;', '>')
        content = content.replace('&amp;', '&')
        content = content.replace('&quot;', '"')
        
        return content.strip()

class NaverSearchAPICrawler(BaseCrawler):
    def __init__(self, keywords, max_pages=1000, save_dir="data/raw"):
        super().__init__(keywords, max_pages=max_pages, save_dir=save_dir)
        self.targets = ["blog", "news", "cafearticle"]

        self.client_id = os.getenv("NAVER_CLIENT_ID")
        self.client_secret = os.getenv("NAVER_CLIENT_SECRET")

        if not self.client_id or not self.client_secret:
            self.logger.error("네이버 API 키가 없습니다.")
            raise ValueError("네이버 API 키가 없습니다.")

        # 로깅 레벨 설정
        self.logger.setLevel(logging.INFO)

        self.headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret
        }
        self.cache = JsonCache()
        self.blog_crawler = NaverBlogCrawler()
        self.doc_ids = set()  # 문서 ID 중복 체크를 위한 세트

        # 원본 키워드 객체 저장
        self.original_keywords = keywords
        # 필터링용 키워드 문자열 추출
        self.keywords = [k['text'] for k in keywords]
        
    def __del__(self):
        if hasattr(self, 'blog_crawler') and self.blog_crawler:
            self.blog_crawler.close_driver()

    def _postprocess(self, items, original_keywords):
        """
        수집된 아이템을 후처리하는 메서드
        
        1. AND/OR 조건에 따라 키워드 필터링
        2. URL+제목 기반 해시로 중복 제거
        """
        processed_items = []
        filtered_count = 0
        
        for item in items:
            # 1. 키워드 조건 확인
            combined_text = f"{item['title']} {item['content']}"
            if not check_keyword_conditions(combined_text, original_keywords):
                self.logger.debug(f"키워드 필터링: 조건에 맞지 않음 - {item['title'][:30]}...")
                filtered_count += 1
                continue
                
            # 2. 중복 문서 확인
            doc_id = hashlib.sha256((item['url'] + item['title']).encode()).hexdigest()
            if doc_id in self.doc_ids:
                self.logger.debug(f"중복 문서: {item['title'][:30]}...")
                filtered_count += 1
                continue  # 예외 대신 스킵
            
            # 문서 ID 추가
            self.doc_ids.add(doc_id)
            item['doc_id'] = doc_id
            
            processed_items.append(item)
        
        # 필터링 비율 계산
        if items:
            filter_ratio = (filtered_count / len(items)) * 100
            self.logger.info(f"필터링 비율: {filter_ratio:.1f}% ({filtered_count}/{len(items)})")
        
        return processed_items

    def generate_keyword_variations(self, keywords):
        """키워드 조건에 따라 검색 쿼리 조합 생성"""
        if not keywords:
            return []
            
        # 지역 키워드 (항상 AND)
        region_keyword = keywords[0]['text'] if keywords else None
        if not region_keyword:
            return []
            
        # AND 키워드 (항상 포함)
        and_keywords = [k['text'] for k in keywords[1:] if k['condition'] == 'AND' and k['text']]
        
        # OR 키워드 (다양한 조합 생성)
        or_keywords = [k['text'] for k in keywords[1:] if k['condition'] == 'OR' and k['text']]
        
        variations = []
        
        # AND 키워드가 있는 경우, 무조건 포함
        if and_keywords:
            # 기본 검색어 (지역 + AND 키워드 모두 포함)
            base_query = f"{region_keyword} {' '.join(and_keywords)}"
            
            # OR 키워드가 없으면 기본 검색어만 사용
            if not or_keywords:
                variations.append(base_query)
            else:
                # OR 키워드가 있으면 각각 조합 추가
                for or_kw in or_keywords:
                    variations.append(f"{base_query} {or_kw}")
                
                # 모든 OR 키워드 한번에 추가 (선택적)
                if len(or_keywords) > 1:
                    variations.append(f"{base_query} {' '.join(or_keywords)}")
        else:
            # AND 키워드가 없는 경우
            # 지역 키워드만 사용
            variations.append(region_keyword)
            
            # OR 키워드가 있으면 각각 조합
            for or_kw in or_keywords:
                variations.append(f"{region_keyword} {or_kw}")
        
        # 중복 제거 후 반환
        return list(set(variations))

    def direct_crawl_naver_search(self, keyword, num_pages=5):
        """
        네이버 검색을 직접 크롤링하는 방법 (API의 한계 보완)
        """
        # 초기화
        results = []
        seen_urls = set()
        
        try:
            # 필요시 드라이버 초기화
            if not hasattr(self.blog_crawler, 'driver') or not self.blog_crawler.driver:
                self.blog_crawler.init_driver()
            
            driver = self.blog_crawler.driver
            if not driver:
                self.logger.error("직접 크롤링을 위한 드라이버 초기화 실패")
                return results
            
            # 검색 URL 생성
            encoded_keyword = quote(keyword)
            
            # 페이지별 크롤링
            for page in range(1, num_pages + 1):
                try:
                    start_idx = (page - 1) * 10 + 1
                    search_url = f"https://search.naver.com/search.naver?where=view&query={encoded_keyword}&start={start_idx}"
                    
                    driver.get(search_url)
                    time.sleep(random.uniform(0.3, 1.0))  # 차단 방지를 위한 대기
                    
                    # 최신 Selenium API 사용
                    from selenium.webdriver.common.by import By
                    
                    # 검색 결과 항목 찾기
                    items = driver.find_elements(By.CSS_SELECTOR, "li.bx._svp_item")
                    
                    if not items:
                        self.logger.info(f"직접 크롤링: 페이지 {page}에서 결과를 찾을 수 없음")
                        break
                    
                    self.logger.info(f"직접 크롤링: 페이지 {page}에서 {len(items)}개 항목 발견")
                    
                    for item in items:
                        try:
                            # 제목 및 링크 추출
                            title_elem = item.find_element(By.CSS_SELECTOR, "a.api_txt_lines.total_tit")
                            title = title_elem.text.strip()
                            url = title_elem.get_attribute("href")
                            
                            # 중복 URL 건너뛰기
                            if url in seen_urls:
                                continue
                            seen_urls.add(url)
                            
                            # 블로그 이름, 날짜 추출
                            blog_name = ""
                            date_str = ""
                            info_elems = item.find_elements(By.CSS_SELECTOR, "span.etc_dsc_area > span")
                            for info in info_elems:
                                text = info.text.strip()
                                if "전" in text or "-" in text or "." in text:  # 날짜 형식 감지
                                    date_str = text
                                else:
                                    blog_name = text
                                    
                            # 내용 추출
                            desc = ""
                            desc_elem = item.find_element(By.CSS_SELECTOR, "div.api_txt_lines.dsc_txt")
                            if desc_elem:
                                desc = desc_elem.text.strip()
                                
                            # 전체 본문 가져오기
                            full_content = self.blog_crawler.get_blog_content(url)
                            content = full_content if full_content else desc
                            
                            # 날짜 형식 변환
                            try:
                                if "." in date_str:
                                    date_parts = date_str.split(".")
                                    if len(date_parts) >= 3:
                                        date_obj = datetime.strptime(f"{date_parts[0]}{date_parts[1]:0>2}{date_parts[2]:0>2}", "%Y%m%d")
                                    else:
                                        date_obj = datetime.now()
                                else:
                                    date_obj = datetime.now()
                            except:
                                date_obj = datetime.now()
                                
                            # 결과 저장
                            post_data = {
                                "title": title,
                                "content": content,
                                "url": url,
                                "blog_name": blog_name,
                                "published_date": date_obj.strftime("%Y%m%d"),
                                "date_obj": date_obj.isoformat(),
                                "platform": "naver_direct_crawl",
                                "keyword": keyword,
                                "original_keywords": ",".join([k['text'] for k in self.original_keywords]),
                                "sentiment": None,
                                "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            }
                            
                            results.append(post_data)
                            self.cache.save(url)
                            
                        except Exception as e:
                            self.logger.warning(f"직접 크롤링 중 항목 처리 오류: {str(e)}")
                            continue
                            
                    # 다음 페이지 버튼이 없거나 비활성화된 경우 종료
                    next_buttons = driver.find_elements(By.CSS_SELECTOR, "a.btn_next")
                    if not next_buttons or "disabled" in next_buttons[0].get_attribute("class"):
                        break
                        
                    time.sleep(random.uniform(0.3, 0.8))  # 차단 방지를 위한 대기
                    
                except Exception as e:
                    self.logger.error(f"직접 크롤링 중 페이지 처리 오류: {str(e)}")
                    break
                
        except Exception as e:
            self.logger.error(f"직접 크롤링 중 오류 발생: {str(e)}")
        
        return results

    def crawl(self):
        combined_results = []
        direct_crawl_results = []
        
        try:
            # 크롤링 시작 시간 기록
            start_time = time.time()
            self.logger.info(f"====== 크롤링 시작: {time.strftime('%Y-%m-%d %H:%M:%S')} ======")
            keyword_info = [f"{k['text']}({'필수' if k.get('condition') == 'AND' else '선택적'})" for k in self.original_keywords]
            self.logger.info(f"검색 키워드: {keyword_info}")
            
            # 1. API 기반 크롤링
            for target in self.targets:
                self.logger.info(f"\n===== 네이버 {target} API 수집 시작 =====")
                api_url = f"https://openapi.naver.com/v1/search/{target}.json"

                keyword_variations = self.generate_keyword_variations(self.original_keywords)
                self.logger.info(f"생성된 검색 쿼리 ({len(keyword_variations)}개): {keyword_variations}")

                for i, keyword in enumerate(keyword_variations, 1):
                    self.logger.info(f"\n>>> 검색 쿼리 {i}/{len(keyword_variations)}: '{keyword}' 처리 중...")
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
                        time.sleep(random.uniform(0.2, 0.7))

                    # 키워드 조건에 따른 필터링
                    filtered_results = self._postprocess(keyword_results, self.original_keywords)
 
                    self.logger.info(f"[SUMMARY] Found {len(keyword_results)} items, filtered to {len(filtered_results)} for '{keyword}'")
                    keyword_results = filtered_results
                    keyword_results.sort(key=lambda x: x['date_obj'], reverse=True)

                    if keyword_results:
                        keywords_str = '_'.join(self.keywords)
                        filename = f"naver_{target}_{len(keyword_results)}_{keywords_str}_{time.strftime('%Y%m%d_%H%M%S')}.json"
                        filepath = os.path.join(self.save_dir, filename)
                        os.makedirs(self.save_dir, exist_ok=True)
                        with open(filepath, "w", encoding="utf-8") as f:
                            json.dump(keyword_results, f, ensure_ascii=False, indent=2)
                        self.logger.info(f"Saved results to {filepath}")
                        combined_results.extend(keyword_results)
                    else:
                        self.logger.warning(f"No data saved for keyword '{keyword}'")
                    
            # 2. 직접 크롤링 (API 결과가 부족하거나 질이 낮을 경우)
            min_acceptable_results = 10  # 최소 허용 결과 수
            min_content_length = 200  # 최소 컨텐츠 길이
            
            # API 결과의 수량과 품질 평가
            need_direct_crawl = False
            
            # 결과 수가 최소 허용치보다 적으면 직접 크롤링 필요
            if len(combined_results) < min_acceptable_results:
                need_direct_crawl = True
                self.logger.info(f"API 결과가 부족하여 직접 크롤링 필요: {len(combined_results)}개 < {min_acceptable_results}개")
            
            # 결과의 품질이 낮으면 직접 크롤링 필요
            else:
                content_lengths = [len(item.get('content', '')) for item in combined_results]
                avg_content_length = sum(content_lengths) / len(content_lengths) if content_lengths else 0
                
                if avg_content_length < min_content_length:
                    need_direct_crawl = True
                    self.logger.info(f"API 결과 품질이 낮아 직접 크롤링 필요: 평균 길이 {avg_content_length:.1f} < {min_content_length}")
            
            if need_direct_crawl:
                self.logger.info("직접 크롤링 시작")
                
                # 주요 키워드 조합으로 직접 크롤링
                main_variations = self.generate_keyword_variations(self.original_keywords)
                if len(main_variations) > 3:
                    main_variations = main_variations[:3]  # 상위 3개 조합만 사용
                
                for keyword in main_variations:
                    self.logger.info(f"직접 크롤링 키워드: {keyword}")
                    direct_results = self.direct_crawl_naver_search(keyword, num_pages=3)
                    
                    # 후처리 적용
                    filtered_direct_results = self._postprocess(direct_results, self.original_keywords)
                    self.logger.info(f"직접 크롤링 결과: {len(direct_results)}개 중 {len(filtered_direct_results)}개 필터링됨")
                    
                    if filtered_direct_results:
                        direct_crawl_results.extend(filtered_direct_results)
                    
                if direct_crawl_results:
                    self.logger.info(f"직접 크롤링으로 {len(direct_crawl_results)}개 문서 추가 수집")
                    keywords_str = '_'.join(self.keywords)
                    filename = f"naver_direct_{len(direct_crawl_results)}_{keywords_str}_{time.strftime('%Y%m%d_%H%M%S')}.json"
                    filepath = os.path.join(self.save_dir, filename)
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(direct_crawl_results, f, ensure_ascii=False, indent=2)
                    self.logger.info(f"Saved direct crawl results to {filepath}")
                    combined_results.extend(direct_crawl_results)

            # 최종 결과 저장
            self.logger.info(f"[FINAL SUMMARY] Total {len(combined_results)} items collected for all keywords")
            if combined_results:
                unique_results = {item['url']: item for item in combined_results}.values()
                combined_results = list(unique_results)
                combined_results.sort(key=lambda x: x['date_obj'], reverse=True)

                keywords_str = '_'.join(self.keywords)
                filename = f"naver_combined_{len(combined_results)}_{keywords_str}_{time.strftime('%Y%m%d_%H%M%S')}.json"
                filepath = os.path.join(self.save_dir, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(combined_results, f, ensure_ascii=False, indent=2)
                self.logger.info(f"Saved combined results to {filepath}")
                
                # 크롤링 종료 시간 기록 및 요약
                end_time = time.time()
                elapsed_time = end_time - start_time
                self.logger.info(f"크롤링 종료: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                self.logger.info(f"소요 시간: {elapsed_time:.2f}초 ({elapsed_time/60:.2f}분)")
                self.logger.info(f"수집된 총 문서: {len(combined_results)}개")
            else:
                self.logger.warning("No data saved (empty result set)")

        except Exception as e:
            self.logger.error(f"Crawler error: {str(e)}")

        return combined_results