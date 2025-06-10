import os
import requests
from bs4 import BeautifulSoup
import time
import random
import json
import logging
import hashlib
from urllib.parse import quote, urljoin, parse_qs, urlparse
from datetime import datetime
from selenium import webdriver

from core.base_crawler import BaseCrawler
from utils.savers import DataSaver
from core.sentiment_analysis_ensemble import EnsembleSentimentAnalyzer

class DCInsideCrawler(BaseCrawler):
    """디시인사이드 크롤러 - 갤러리 게시글 및 댓글 수집"""
    
    def __init__(self, keywords, max_pages=5, max_comments=30, save_dir="data/raw", analyze_sentiment=True, respect_robots=True, browser_type="chrome"):
        """
        DCInside 크롤러 초기화
        
        Args:
            keywords: 검색할 키워드 리스트
            max_pages: 수집할 최대 페이지 수
            max_comments: 각 게시글당 수집할 최대 댓글 수
            save_dir: 저장 디렉터리
            analyze_sentiment: 감성 분석 수행 여부
            respect_robots: robots.txt 정책 준수 여부 (True: 준수, False: 무시)
            browser_type: 사용할 브라우저 타입 ("chrome" 또는 "firefox")
        """
        super().__init__(keywords, max_pages, save_dir)
        self.base_url = "https://search.dcinside.com/post/p"
        self.post_base_url = "https://gall.dcinside.com"
        self.max_comments = max_comments
        self.doc_ids = set()
        self.analyze_sentiment = analyze_sentiment
        self.respect_robots = respect_robots
        self.browser_type = browser_type
        
        # 필터링 조건
        self.filter_conditions = {
            'min_content_length': 50,
            'max_pages': max_pages,
            'min_confidence': 0.0,
            'exclude_keywords': ['광고', '홍보', 'sponsored', '출처', '저작권'],
            'required_keywords': ['부안'],
            'date_range': None,
            'include_notice': True
        }
        
        # HTTP 헤더 설정
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Referer': 'https://www.dcinside.com/'
        }
        
        # Selenium 옵션 설정
        if browser_type == "chrome":
            from selenium.webdriver.chrome.options import Options
            options = Options()
        elif browser_type == "firefox":
            from selenium.webdriver.firefox.options import Options
            options = Options()
        else:
            raise ValueError(f"지원되지 않는 브라우저 타입: {browser_type}")
            
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-notifications")
        
        self.options = options
        
        # 감성 분석기 초기화 (지연 로딩)
        self.sentiment_analyzer = None
        
        # 로깅 설정
        self.logger.setLevel(logging.INFO)
        
        # 키워드 설정
        if isinstance(keywords, list) and all(isinstance(k, dict) for k in keywords):
            self.original_keywords = keywords
            self.keywords = [k['text'] for k in keywords]
        else:
            self.keywords = keywords if isinstance(keywords, list) else [keywords]
            self.original_keywords = [{'text': k, 'condition': 'AND'} for k in self.keywords]
            
        # robots.txt 정책 확인 및 경고
        if self.respect_robots:
            self.logger.warning("""
            ⚠️ DCinside robots.txt 주의사항 ⚠️
            DCinside의 robots.txt는 일반 크롤러의 접근을 전체적으로 차단하고 있습니다(User-agent: * / Disallow: /).
            검색 엔진 봇(Googlebot, Yeti 등)만 접근을 허용하고 있습니다.
            이 크롤러는 기본적으로 robots.txt 정책을 준수하도록 설정되어 있어서 작동하지 않을 것입니다.
            크롤링을 수행하려면 respect_robots=False로 설정하세요.
            웹사이트 정책을 위반하면 법적 문제가 발생할 수 있으니 주의하세요.
            """)
    
    def _search_posts(self, keyword, page=1):
        """검색 키워드로 게시글 검색"""
        try:
            # 검색 URL 생성
            encoded_keyword = quote(keyword)
            search_url = f"{self.base_url}?q={encoded_keyword}&p={page}"
            
            # 요청
            response = requests.get(
                search_url,
                headers=self.headers,
                timeout=10
            )
            
            # 🔍 Save HTML for debugging
            debug_path = os.path.join("debug", f"dcinside_search_{keyword}_{page}.html")
            os.makedirs(os.path.dirname(debug_path), exist_ok=True)
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            self.logger.info(f"🔍 검색 HTML 저장됨: {debug_path}")
            
            # 응답 확인
            if response.status_code != 200:
                self.logger.error(f"검색 요청 실패: {response.status_code}")
                return []
                
            # 파싱
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 검색 결과 없음 확인
            no_result = soup.select_one('.search_no_data')
            if no_result:
                self.logger.info(f"검색 결과가 없습니다: {keyword}")
                return []
            
            # 게시글 목록 추출 (최신 선택자로 변경)
            posts = []
            post_items = soup.select('ul.sch_result_list > li')
            
            if not post_items:
                self.logger.warning(f"검색 결과 파싱 실패: {keyword} (페이지 {page})")
                return []
            
            self.logger.info(f"검색 결과 {len(post_items)}개 발견")
            
            for item in post_items:
                try:
                    # 제목 추출 (최신 선택자로 변경)
                    title_el = item.select_one('a.sch_tit')
                    if not title_el:
                        continue
                        
                    title = self._clean_text(title_el.get_text())
                    
                    # URL 추출 (이미 절대 경로)
                    post_url = title_el.get('href')
                    if not post_url:
                        continue
                        
                    # 작성자 추출 (최신 선택자로 변경)
                    author_el = item.select_one('.user_nick')
                    author = self._clean_text(author_el.get_text()) if author_el else "Unknown"
                    
                    # 작성일 추출 (최신 선택자로 변경)
                    date_el = item.select_one('.date')
                    pub_date = self._clean_text(date_el.get_text()) if date_el else None
                    
                    if title and post_url:
                        posts.append({
                            'title': title,
                            'url': post_url,
                            'published_date': pub_date,
                            'author': author
                        })
                        self.logger.debug(f"게시글 파싱 성공: {title}")
                except Exception as e:
                    self.logger.error(f"게시글 항목 파싱 중 오류: {str(e)}")
                    continue
                    
            return posts
            
        except Exception as e:
            self.logger.error(f"검색 중 오류: {str(e)}")
            return []

    def _get_post_content(self, post_url):
        """게시글 본문 가져오기"""
        try:
            response = requests.get(post_url, headers=self.headers, timeout=10)
            if response.status_code != 200:
                return None
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 제목
            title_el = soup.select_one('.title_subject')
            title = self._clean_text(title_el.get_text()) if title_el else ""
            
            # 본문
            content_el = soup.select_one('div.write_div')
            content = self._clean_text(content_el.get_text()) if content_el else ""
            
            # 작성자
            author_el = soup.select_one('span.nickname')
            author = self._clean_text(author_el.get_text()) if author_el else "Unknown"
            
            # 작성일
            date_el = soup.select_one('span.gall_date')
            pub_date = self._clean_text(date_el.get('title') or date_el.get_text()) if date_el else None
            
            # 댓글
            comments = self._get_comments(post_url)
            
            return {
                'title': title,
                'content': content,
                'author': author,
                'published_date': pub_date,
                'comments': comments,
                'url': post_url
            }
            
        except Exception as e:
            self.logger.error(f"게시글 본문 가져오기 중 오류: {str(e)}")
            return None

    def crawl(self):
        """디시인사이드 데이터 수집"""
        try:
            start_time = time.time()
            self.logger.info(f"====== 크롤링 시작: {time.strftime('%Y-%m-%d %H:%M:%S')} ======")
            
            all_posts = []
            
            for keyword in self.keywords:
                self.logger.info(f"\n===== 키워드 '{keyword}' 검색 시작 =====")
                
                for page in range(1, self.filter_conditions['max_pages'] + 1):
                    self.logger.info(f"페이지 {page} 처리 중...")
                    posts = self._search_posts(keyword, page)
                    
                    if not posts:
                        break
                        
                    for post in posts:
                        try:
                            post_details = self._get_post_content(post['url'])
                            if post_details:
                                # 게시글 ID 생성
                                doc_id = hashlib.sha256(post['url'].encode()).hexdigest()
                                
                                if doc_id in self.doc_ids:
                                    continue
                                    
                                self.doc_ids.add(doc_id)
                                
                                # 감성 분석
                                combined_text = f"{post_details['title']} {post_details['content']}"
                                sentiment, confidence = self.analyze_text_sentiment(combined_text)
                                
                                post_data = {
                                    **post_details,
                                    'platform': 'dcinside',
                                    'keyword': keyword,
                                    'original_keywords': ",".join(self.keywords),
                                    'sentiment': sentiment,
                                    'confidence': confidence,
                                    'doc_id': doc_id,
                                    'crawled_at': time.strftime("%Y-%m-%d %H:%M:%S")
                                }
                                
                                all_posts.append(post_data)
                                
                        except Exception as e:
                            self.logger.error(f"게시글 처리 중 오류: {str(e)}")
                            continue
                            
                    time.sleep(random.uniform(0.5, 1.0))
                    
            # 후처리 적용
            filtered_posts = self._postprocess(all_posts, self.original_keywords)
            
            # 결과 저장
            if filtered_posts:
                keywords_str = '_'.join(self.keywords)
                filename = f"dcinside_{len(filtered_posts)}_{keywords_str}_{time.strftime('%Y%m%d_%H%M%S')}.json"
                filepath = os.path.join(self.save_dir, filename)
                os.makedirs(self.save_dir, exist_ok=True)
                
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(filtered_posts, f, ensure_ascii=False, indent=2)
                    
                self.logger.info(f"결과 저장 완료: {filepath}")
                
            # 크롤링 종료 시간 기록 및 요약
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.logger.info(f"크롤링 종료: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"소요 시간: {elapsed_time:.2f}초 ({elapsed_time/60:.2f}분)")
            self.logger.info(f"수집된 총 문서: {len(filtered_posts)}개")
            
            return filtered_posts
            
        except Exception as e:
            self.logger.error(f"크롤링 중 오류 발생: {str(e)}")
            return []

    def _get_comments(self, post_url):
        """게시글 댓글 가져오기"""
        comments = []
        try:
            # URL에서 갤러리 ID와 게시글 번호 추출
            parsed_url = urlparse(post_url)
            query_params = parse_qs(parsed_url.query)
            
            gallery_id = query_params.get('id', [''])[0]
            post_id = query_params.get('no', [''])[0]
            
            if not gallery_id or not post_id:
                return comments
                
            # 댓글 API URL
            comment_url = f"{self.post_base_url}/board/comment/"
            
            data = {
                'id': gallery_id,
                'no': post_id,
                'cmt_id': gallery_id,
                'cmt_no': post_id,
                'e_s_n_o': '3eabc219ebdd65f1'
            }
            
            headers = self.headers.copy()
            headers.update({
                'X-Requested-With': 'XMLHttpRequest',
                'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
                'Origin': self.post_base_url,
                'Referer': post_url
            })
            
            response = requests.post(comment_url, data=data, headers=headers, timeout=10)
            
            if response.status_code == 200:
                try:
                    comment_data = response.json()
                    comment_html = comment_data.get('comments', '')
                    comment_soup = BeautifulSoup(comment_html, 'html.parser')
                    
                    for i, comment_el in enumerate(comment_soup.select('li.ub-content')):
                        if i >= self.max_comments:
                            break
                            
                        try:
                            nick_el = comment_el.select_one('span.nickname')
                            content_el = comment_el.select_one('p.usertxt')
                            date_el = comment_el.select_one('span.date_time')
                            
                            comment = {
                                'author': self._clean_text(nick_el.get_text()) if nick_el else "Unknown",
                                'content': self._clean_text(content_el.get_text()) if content_el else "",
                                'date': self._clean_text(date_el.get_text()) if date_el else "",
                                'sentiment': None,
                                'confidence': None
                            }
                            
                            if self.analyze_sentiment:
                                sentiment, confidence = self.analyze_text_sentiment(comment['content'])
                                comment.update({
                                    'sentiment': sentiment,
                                    'confidence': confidence
                                })
                                
                            comments.append(comment)
                            
                        except Exception as e:
                            self.logger.error(f"댓글 항목 파싱 중 오류: {str(e)}")
                            continue
                            
                except json.JSONDecodeError:
                    self.logger.error("댓글 응답이 유효한 JSON이 아닙니다.")
                    
        except Exception as e:
            self.logger.error(f"댓글 가져오기 중 오류: {str(e)}")
            
        return comments
    
    def _clean_text(self, text):
        """HTML 태그 제거 및 텍스트 정리"""
        if not text:
            return ""
            
        # 기본 정리
        text = super().clean_text(text)
        
        # 특수 문자 처리
        text = text.replace('&nbsp;', ' ')
        text = text.replace('\xa0', ' ')
        text = text.replace('\u200b', '')
        
        return text.strip()
    
    def _postprocess(self, items, original_keywords):
        """수집된 아이템을 후처리하는 메서드"""
        processed_items = []
        filtered_count = 0
        
        for item in items:
            # 1. 키워드 조건 확인
            combined_text = f"{item['title']} {item.get('content', '')}"
            
            # 필수 키워드(부안)는 반드시 포함
            if not any(kw.lower() in combined_text.lower() for kw in self.filter_conditions['required_keywords']):
                filtered_count += 1
                continue
                
            # 제외 키워드가 있으면 제외
            if any(kw.lower() in combined_text.lower() for kw in self.filter_conditions['exclude_keywords']):
                filtered_count += 1
                continue
                
            # 2. 중복 문서 확인
            doc_id = hashlib.sha256((item['url'] + item['title']).encode()).hexdigest()
            if doc_id in self.doc_ids:
                filtered_count += 1
                continue
            
            # 3. 날짜 범위 확인
            try:
                pub_date = datetime.strptime(item['published_date'], '%Y%m%d')
                if (datetime.now() - pub_date).days > self.filter_conditions['date_range']:
                    filtered_count += 1
                    continue
            except:
                pass
            
            # 문서 ID 추가
            self.doc_ids.add(doc_id)
            item['doc_id'] = doc_id
            
            processed_items.append(item)
        
        # 필터링 비율 계산
        if items:
            filter_ratio = (filtered_count / len(items)) * 100
            self.logger.info(f"필터링 비율: {filter_ratio:.1f}% ({filtered_count}/{len(items)})")
        
        return processed_items
        
    def analyze_text_sentiment(self, text):
        """텍스트 감성 분석"""
        if not self.analyze_sentiment or not text:
            return None, None
            
        try:
            if self.sentiment_analyzer is None:
                self.sentiment_analyzer = EnsembleSentimentAnalyzer()
            sentiment, confidence = self.sentiment_analyzer.predict(text)
            return sentiment, confidence
        except Exception as e:
            self.logger.error(f"감성 분석 중 오류: {str(e)}")
            return None, None 