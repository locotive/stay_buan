import os
import time
import random
import json
import logging
import hashlib
import requests
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.chrome.service import Service as ChromeService

from core.base_crawler import BaseCrawler
from core.sentiment_analysis_ensemble import EnsembleSentimentAnalyzer

class BuanGovCrawler(BaseCrawler):
    """부안군청 홈페이지 크롤러 - 공지사항, 보도자료 등 게시판 수집"""
    
    def __init__(self, keywords, max_pages=5, save_dir="data/raw", analyze_sentiment=True, browser_type="chrome"):
        """
        부안군청 크롤러 초기화
        
        Args:
            keywords: 검색할 키워드 리스트
            max_pages: 수집할 최대 페이지 수
            save_dir: 저장 디렉터리
            analyze_sentiment: 감성 분석 수행 여부
            browser_type: 사용할 브라우저 타입 ("chrome" 또는 "firefox")
        """
        super().__init__(keywords, max_pages, save_dir)
        self.base_url = "https://www.buan.go.kr"
        self.browser_type = browser_type
        
        # 게시판 설정 개선 - 선택자와 대기 조건을 포함
        self.boards = [
            {
                "name": "공지사항",
                "url": "/board/list.buan?boardId=BBS_0000002&listPage=true",
                "id": "BBS_0000002",
                "selectors": {
                    "list": ".bbs_list table tbody tr",
                    "title": "td.title a, td:nth-child(2) a",
                    "date": "td:nth-child(5), td:nth-child(4)",
                    "author": "td:nth-child(3), td:nth-child(4)",
                    "content": ".bbs_content, .view_contents, .content",
                    "attachments": ".bbs_file a, .view_file a"
                },
                "wait_conditions": {
                    "list": (By.CSS_SELECTOR, ".bbs_list table tbody tr"),
                    "content": (By.CSS_SELECTOR, ".bbs_content, .view_contents, .content")
                }
            },
            {
                "name": "보도자료",
                "url": "/board/list.buanNews",
                "id": "buanNews",
                "selectors": {
                    "list": ".container.sub .table_list tbody tr",
                    "title": "td.title a, td:nth-child(2) a",
                    "date": "td:nth-child(5), td:nth-child(4)",
                    "author": "td:nth-child(3), td:nth-child(4)",
                    "content": ".bbs_content, .view_contents, .content",
                    "attachments": ".bbs_file a, .view_file a"
                },
                "wait_conditions": {
                    "list": (By.CSS_SELECTOR, ".container.sub .table_list tbody tr"),
                    "content": (By.CSS_SELECTOR, ".bbs_content, .view_contents, .content")
                }
            },
            {
                "name": "군정소식",
                "url": "/board/list.buan?boardId=BBS_0000004&listPage=true",
                "id": "BBS_0000004"
            },
            {
                "name": "고시공고",
                "url": "/board/list.buan?boardId=BBS_0000005&listPage=true",
                "id": "BBS_0000005"
            },
            {
                "name": "자유게시판",
                "url": "/board/list.buan?boardId=BBS_0000006&listPage=true",
                "id": "BBS_0000006"
            },
            {
                "name": "칭찬합니다",
                "url": "/board/list.buan?boardId=BBS_0000007&listPage=true",
                "id": "BBS_0000007"
            },
            {
                "name": "소통광장",
                "url": "/board/list.buan?boardId=BBS_0000008&listPage=true",
                "id": "BBS_0000008"
            }
        ]
        
        self.doc_ids = set()  # 중복 문서 확인용
        self.analyze_sentiment = analyze_sentiment
        
        # Selenium 옵션 설정
        if browser_type == "chrome":
            from selenium.webdriver.chrome.options import Options
            options = Options()
        elif browser_type == "firefox":
            from selenium.webdriver.firefox.options import Options
            options = Options()
        else:
            raise ValueError(f"지원되지 않는 브라우저 타입: {browser_type}")
            
        options.add_argument("--headless")  # GUI 없이 실행
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-notifications")
        
        self.options = options
        
        # 감성 분석기 초기화 (지연 로딩)
        self.sentiment_analyzer = None
        
        # 로깅 레벨 설정
        self.logger.setLevel(logging.INFO)
        
        # 원본 키워드 객체 저장
        if isinstance(keywords, list) and all(isinstance(k, dict) for k in keywords):
            self.original_keywords = keywords
            # 필터링용 키워드 문자열 추출
            self.keywords = [k['text'] for k in keywords]
        else:
            # 문자열 리스트인 경우 형식 변환
            self.keywords = keywords if isinstance(keywords, list) else [keywords]
            self.original_keywords = [{'text': k, 'condition': 'AND'} for k in self.keywords]
        
        # 첨부파일 저장 디렉토리
        self.attachment_dir = os.path.join(save_dir, "attachments")
        os.makedirs(self.attachment_dir, exist_ok=True)
        
        # 동적 대기 시간 설정
        self.wait_config = {
            "min_wait": 0.5,  # 최소 대기 시간
            "max_wait": 2.0,  # 최대 대기 시간
            "timeout": 10,    # 요소 대기 타임아웃
            "retry_count": 3  # 재시도 횟수
        }
    
    def _get_sentiment_analyzer(self):
        """감성 분석기 지연 로딩"""
        if self.sentiment_analyzer is None:
            self.sentiment_analyzer = EnsembleSentimentAnalyzer()
        return self.sentiment_analyzer
    
    def analyze_text_sentiment(self, text):
        """텍스트 감성 분석"""
        if not self.analyze_sentiment or not text:
            return None, None
            
        try:
            analyzer = self._get_sentiment_analyzer()
            sentiment, confidence = analyzer.predict(text)
            return sentiment, confidence
        except Exception as e:
            self.logger.error(f"감성 분석 중 오류: {str(e)}")
            return None, None
    
    def init_driver(self):
        """웹드라이버 초기화 - Docker 환경 고려"""
        try:
            if self.browser_type == "chrome":
                # Chrome 옵션 설정
                chrome_options = webdriver.ChromeOptions()
                
                # 필수 headless 옵션
                chrome_options.add_argument('--headless=new')  # 새로운 headless 모드
                chrome_options.add_argument('--no-sandbox')
                chrome_options.add_argument('--disable-dev-shm-usage')
                chrome_options.add_argument('--disable-gpu')
                chrome_options.add_argument('--window-size=1920,1080')
                chrome_options.add_argument('--disable-extensions')
                chrome_options.add_argument('--disable-notifications')
                
                # Docker 환경 감지 및 드라이버 경로 설정
                if os.getenv('DOCKER_ENV'):
                    self.logger.info("Docker 환경에서 실행 중")
                    driver_path = "/usr/bin/chromium-driver"
                    # Docker 환경에서 필요한 추가 옵션
                    chrome_options.add_argument('--disable-software-rasterizer')
                    chrome_options.add_argument('--disable-setuid-sandbox')
                    chrome_options.binary_location = "/usr/bin/chromium"
                else:
                    self.logger.info("로컬 환경에서 실행 중")
                    driver_path = "chromedriver"
                
                try:
                    # Chrome 서비스 설정
                    chrome_service = ChromeService(executable_path=driver_path)
                    self.logger.info(f"Chrome 드라이버 경로: {driver_path}")
                    
                    # 드라이버 초기화
                    driver = webdriver.Chrome(
                        service=chrome_service,
                        options=chrome_options
                    )
                    
                    # 타임아웃 설정
                    driver.set_page_load_timeout(30)
                    driver.set_script_timeout(30)
                    
                    self.logger.info("Chrome 드라이버 초기화 성공")
                    return driver
                    
                except Exception as e:
                    self.logger.error(f"Chrome 드라이버 초기화 실패: {str(e)}")
                    self.logger.error("상세 오류 정보:", exc_info=True)
                    return None
                
            elif self.browser_type == "firefox":
                # Firefox 옵션 설정
                firefox_options = webdriver.FirefoxOptions()
                firefox_options.add_argument('--headless')
                firefox_options.add_argument('--width=1920')
                firefox_options.add_argument('--height=1080')
                
                try:
                    driver = webdriver.Firefox(options=firefox_options)
                    driver.set_page_load_timeout(30)
                    self.logger.info("Firefox 드라이버 초기화 성공")
                    return driver
                except Exception as e:
                    self.logger.error(f"Firefox 드라이버 초기화 실패: {str(e)}")
                    self.logger.error("상세 오류 정보:", exc_info=True)
                    return None
                
            else:
                self.logger.error(f"지원하지 않는 브라우저 타입: {self.browser_type}")
                return None
            
        except Exception as e:
            self.logger.error(f"드라이버 초기화 중 예상치 못한 오류: {str(e)}")
            self.logger.error("상세 오류 정보:", exc_info=True)
            return None
    
    def _download_attachment(self, url, filename):
        """첨부파일 다운로드"""
        try:
            # 파일 경로 생성
            filepath = os.path.join(self.attachment_dir, filename)
            
            # 이미 다운로드된 파일이면 스킵
            if os.path.exists(filepath):
                return filepath
                
            # 다운로드
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # 파일 저장
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        
            return filepath
            
        except Exception as e:
            self.logger.error(f"첨부파일 다운로드 실패: {url} - {str(e)}")
            return None

    def _wait_for_element(self, driver, condition, timeout=None):
        """요소가 나타날 때까지 동적으로 대기"""
        if timeout is None:
            timeout = self.wait_config['timeout']
            
        try:
            element = WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located(condition)
            )
            return element
        except TimeoutException:
            self.logger.warning(f"요소 대기 시간 초과: {condition}")
            return None

    def _get_post_details(self, driver, post_url, board_config):
        """게시글 상세 정보 수집 (개선된 버전)"""
        try:
            # 페이지 로드
            driver.get(post_url)
            
            # 동적 대기 시간 적용
            time.sleep(random.uniform(self.wait_config['min_wait'], self.wait_config['max_wait']))
            
            # 본문 내용 대기 및 추출
            content_el = self._wait_for_element(driver, board_config['wait_conditions']['content'])
            if not content_el:
                return None
                
            content = content_el.text.strip()
            
            # 첨부파일 처리
            attachments = []
            attach_elements = driver.find_elements(By.CSS_SELECTOR, board_config['selectors']['attachments'])
            
            for attach in attach_elements:
                try:
                    file_url = attach.get_attribute("href")
                    if not file_url:
                        continue
                        
                    # 파일명 추출
                    filename = attach.text.strip() or os.path.basename(file_url)
                    if not filename:
                        continue
                        
                    # 다운로드
                    local_path = self._download_attachment(file_url, filename)
                    if local_path:
                        attachments.append({
                            "original_name": filename,
                            "local_path": local_path,
                            "url": file_url
                        })
                except Exception as e:
                    self.logger.error(f"첨부파일 처리 중 오류: {str(e)}")
            
            # 작성일 재확인
            date_el = driver.find_elements(By.CSS_SELECTOR, ".bbs_view th:contains('등록일'), .view_info span:contains('작성일')")
            published_date = None
            if date_el:
                try:
                    date_text = date_el[0].find_element(By.XPATH, "following-sibling::*").text.strip()
                    published_date = self._format_date(date_text)
                except:
                    pass
            
            return {
                "content": content,
                "attachments": attachments,
                "published_date": published_date
            }
            
        except Exception as e:
            self.logger.error(f"게시글 상세 정보 추출 중 오류: {str(e)}")
            return None

    def _search_board(self, driver, board, keyword, page=1):
        """게시판 검색 (개선된 버전)"""
        try:
            # 게시판 URL 생성
            board_url = f"{self.base_url}{board['url']}"
            if "?" in board_url:
                search_url = f"{board_url}&searchKeyword={keyword}&currentPage={page}"
            else:
                search_url = f"{board_url}?searchKeyword={keyword}&currentPage={page}"
            
            self.logger.info(f"게시판 {board['name']} 검색 URL: {search_url}")
            
            # 페이지 로드
            driver.get(search_url)
            
            # 동적 대기 적용
            time.sleep(random.uniform(self.wait_config['min_wait'], self.wait_config['max_wait']))
            
            # 게시글 목록 대기
            list_elements = self._wait_for_element(driver, board['wait_conditions']['list'])
            if not list_elements:
                return []
            
            # 게시글 목록 추출
            posts = []
            article_elements = driver.find_elements(By.CSS_SELECTOR, board['selectors']['list'])
            
            for article in article_elements:
                try:
                    # 공지사항 스킵
                    if article.find_elements(By.CSS_SELECTOR, ".noti"):
                        continue
                    
                    # 제목 및 링크
                    title_el = article.find_element(By.CSS_SELECTOR, board['selectors']['title'])
                    title = title_el.text.strip()
                    href = title_el.get_attribute("href")
                    
                    # 작성일
                    date_el = article.find_elements(By.CSS_SELECTOR, board['selectors']['date'])
                    date_text = date_el[0].text.strip() if date_el else ""
                    
                    # 작성자
                    author_el = article.find_elements(By.CSS_SELECTOR, board['selectors']['author'])
                    author = author_el[0].text.strip() if author_el else "부안군"
                    
                    if title and href:
                        posts.append({
                            "title": title,
                            "url": href,
                            "published_date": date_text,
                            "author": author,
                            "board": board['name']
                        })
                except Exception as e:
                    self.logger.error(f"게시글 항목 파싱 중 오류: {str(e)}")
                    continue
            
            return posts
            
        except Exception as e:
            self.logger.error(f"게시판 검색 중 오류: {str(e)}")
            return []
    
    def _format_date(self, date_text):
        """날짜 텍스트를 표준 형식으로 변환"""
        try:
            date_text = date_text.strip()
            
            # 다양한 날짜 형식 처리
            if "." in date_text:
                parts = date_text.split(".")
                if len(parts) >= 3:
                    year = parts[0].strip()
                    month = parts[1].strip().zfill(2)
                    day = parts[2].strip().zfill(2)
                    return f"{year}{month}{day}"
            elif "-" in date_text:
                parts = date_text.split("-")
                if len(parts) >= 3:
                    year = parts[0].strip()
                    month = parts[1].strip().zfill(2)
                    day = parts[2].strip().zfill(2)
                    return f"{year}{month}{day}"
            elif "/" in date_text:
                parts = date_text.split("/")
                if len(parts) >= 3:
                    year = parts[0].strip()
                    month = parts[1].strip().zfill(2)
                    day = parts[2].strip().zfill(2)
                    return f"{year}{month}{day}"
            
            # 다른 형식은 현재 날짜 반환
            return time.strftime("%Y%m%d")
                
        except Exception as e:
            self.logger.error(f"날짜 형식 변환 중 오류: {str(e)}")
            return time.strftime("%Y%m%d")
    
    def _check_keyword_match(self, text, keywords):
        """텍스트에 키워드가 포함되어 있는지 확인"""
        if not text or not keywords:
            return False
        
        # AND 키워드와 OR 키워드 추출
        and_keywords = [k['text'].lower() for k in keywords if k['condition'] == 'AND' and k['text']]
        or_keywords = [k['text'].lower() for k in keywords if k['condition'] == 'OR' and k['text']]
        
        # AND 키워드는 하나라도 포함되어야 함
        if and_keywords:
            and_found = False
            for kw in and_keywords:
                if kw in text.lower():
                    and_found = True
                    break
            if not and_found:
                return False
        
        # OR 키워드는 하나라도 포함되면 통과
        if or_keywords:
            or_found = False
            for kw in or_keywords:
                if kw in text.lower():
                    or_found = True
                    break
            if not or_found and and_keywords:  # AND 키워드가 있으면 OR 키워드도 하나는 포함되어야 함
                return False
        
        return True
    
    def crawl(self):
        """부안군청 홈페이지 데이터 수집"""
        all_results = []
        driver = None
        
        try:
            # 크롤링 시작 시간 기록
            start_time = time.time()
            self.logger.info(f"====== 부안군청 홈페이지 크롤링 시작: {time.strftime('%Y-%m-%d %H:%M:%S')} ======")
            
            if isinstance(self.original_keywords[0], dict):
                keyword_info = [f"{k['text']}({'필수' if k.get('condition') == 'AND' else '선택적'})" for k in self.original_keywords]
                self.logger.info(f"검색 키워드: {keyword_info}")
            else:
                self.logger.info(f"검색 키워드: {self.keywords}")
            
            # 웹드라이버 초기화
            driver = self.init_driver()
            if not driver:
                self.logger.error("웹드라이버 초기화 실패, 크롤링을 중단합니다.")
                return []
            
            # 각 게시판별로 크롤링
            for board in self.boards:
                self.logger.info(f"\n=== '{board['name']}' 게시판 처리 중... ===")
                board_results = []
                
                # 키워드별 검색
                for keyword in self.keywords:
                    self.logger.info(f"키워드 '{keyword}' 검색 중...")
                    
                    # 여러 페이지 처리
                    for page in range(1, self.max_pages + 1):
                        self.logger.info(f"페이지 {page}/{self.max_pages} 처리 중...")
                        
                        # 검색으로 게시글 목록 가져오기
                        posts = self._search_board(driver, board, keyword, page)
                        
                        if not posts:
                            self.logger.info(f"키워드 '{keyword}' 페이지 {page}에서 게시글을 찾을 수 없습니다.")
                            break
                            
                        self.logger.info(f"페이지 {page}에서 {len(posts)}개 게시글 발견")
                        
                        # 게시글별 상세 정보 가져오기
                        for i, post in enumerate(posts):
                            try:
                                self.logger.info(f"게시글 {i+1}/{len(posts)}: '{post['title'][:20]}...' 처리 중")
                                
                                # 상세 정보 가져오기
                                post_details = self._get_post_details(driver, post['url'], board)
                                
                                if not post_details:
                                    self.logger.warning(f"게시글 상세 정보를 가져올 수 없습니다: {post['url']}")
                                    continue
                                    
                                # 게시글 정보 통합
                                if post_details.get('published_date'):
                                    post['published_date'] = post_details['published_date']
                                
                                post['content'] = post_details.get('content', '')
                                post['attachments'] = post_details.get('attachments', [])
                                
                                # 날짜 형식 통일
                                formatted_date = self._format_date(post['published_date'])
                                post['published_date'] = formatted_date
                                
                                # 키워드 매칭 확인
                                combined_text = f"{post['title']} {post['content']}"
                                if not self._check_keyword_match(combined_text, self.original_keywords):
                                    self.logger.debug(f"키워드 매칭 실패: {post['title'][:20]}...")
                                    continue
                                
                                # 게시글 ID 생성
                                doc_id = hashlib.sha256(f"buan_gov_{post['url']}".encode()).hexdigest()
                                
                                # 중복 확인
                                if doc_id in self.doc_ids:
                                    self.logger.debug(f"중복 게시글 건너뛰기: {post['title'][:20]}...")
                                    continue
                                self.doc_ids.add(doc_id)
                                
                                # 감성 분석
                                sentiment, confidence = self.analyze_text_sentiment(combined_text)
                                
                                # 날짜 객체 생성
                                try:
                                    date_obj = datetime.strptime(formatted_date, "%Y%m%d")
                                except:
                                    date_obj = datetime.now()
                                
                                # 최종 게시글 데이터 구성
                                post_data = {
                                    'doc_id': doc_id,
                                    'title': post['title'],
                                    'content': post['content'],
                                    'url': post['url'],
                                    'author': post.get('author', '부안군'),
                                    'platform': 'buan_gov',
                                    'board': post.get('board', board['name']),
                                    'keyword': keyword,
                                    'original_keywords': ','.join(self.keywords),
                                    'published_date': formatted_date,
                                    'date_obj': date_obj.isoformat(),
                                    'attachments': post.get('attachments', []),
                                    'sentiment': sentiment,
                                    'confidence': confidence,
                                    'crawled_at': time.strftime("%Y-%m-%d %H:%M:%S")
                                }
                                
                                board_results.append(post_data)
                                self.logger.debug(f"게시글 처리 완료: {post['title'][:20]}...")
                                
                                # 요청 간 딜레이
                                time.sleep(random.uniform(1.0, 2.0))
                                
                            except Exception as e:
                                self.logger.error(f"게시글 처리 중 오류: {str(e)}")
                        
                        # 페이지 간 딜레이
                        time.sleep(random.uniform(2.0, 3.0))
                
                # 게시판별 결과 저장
                if board_results:
                    board_name = board['name'].replace(' ', '_')
                    filename = f"buan_gov_{board_name}_{len(board_results)}_{time.strftime('%Y%m%d_%H%M%S')}.json"
                    filepath = os.path.join(self.save_dir, filename)
                    os.makedirs(self.save_dir, exist_ok=True)
                    
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(board_results, f, ensure_ascii=False, indent=2)
                        
                    self.logger.info(f"'{board['name']}' 게시판에서 {len(board_results)}개 게시글 저장 완료: {filepath}")
                    all_results.extend(board_results)
                else:
                    self.logger.warning(f"'{board['name']}' 게시판에서 결과가 없습니다.")
            
            # 크롤링 종료 시간 기록 및 요약
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.logger.info(f"크롤링 종료: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"소요 시간: {elapsed_time:.2f}초 ({elapsed_time/60:.2f}분)")
            self.logger.info(f"수집된 총 게시글: {len(all_results)}개")
            
            # 모든 결과 통합 저장
            if all_results:
                combined_filename = f"buan_gov_combined_{len(all_results)}_{'_'.join(self.keywords)}_{time.strftime('%Y%m%d_%H%M%S')}.json"
                combined_filepath = os.path.join(self.save_dir, combined_filename)
                
                with open(combined_filepath, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)
                    
                self.logger.info(f"통합 결과 저장 완료: {combined_filepath}")
            
        except Exception as e:
            self.logger.error(f"크롤링 중 오류 발생: {str(e)}")
        
        finally:
            # 웹드라이버 종료
            if driver:
                try:
                    driver.quit()
                except:
                    pass
        
        return all_results 