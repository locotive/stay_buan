from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import random
from urllib.parse import quote

from core.base_crawler import BaseCrawler
from utils.savers import DataSaver

class NaverBlogCrawler(BaseCrawler):
    """네이버 블로그 크롤러"""
    
    def __init__(self, keywords, max_pages=3, save_dir="data/raw"):
        super().__init__(keywords, max_pages, save_dir)
        self.base_url = "https://search.naver.com/search.naver"
    
    def setup_driver(self):
        """셀레니움 드라이버 설정"""
        chrome_options = Options()
        
        # 헤드리스 모드 설정 (UI 없이 실행)
        # chrome_options.add_argument("--headless") 
        
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-extensions")
        
        # 사용자 에이전트 설정
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36")
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # 페이지 로딩 타임아웃 설정
        driver.set_page_load_timeout(30)
        
        return driver
    
    def crawl(self):
        """네이버 블로그 크롤링 실행"""
        driver = self.setup_driver()
        all_results = []
        
        try:
            for keyword in self.keywords:
                self.logger.info(f"Crawling for keyword: {keyword}")
                keyword_results = []
                
                for page in range(1, self.max_pages + 1):
                    try:
                        # 시작 위치 계산 (네이버 페이지네이션)
                        start = (page - 1) * 10 + 1
                        
                        # 검색 URL 생성
                        encoded_keyword = quote(keyword)
                        search_url = f"{self.base_url}?where=blog&sm=tab_pge&query={encoded_keyword}&start={start}"
                        
                        self.logger.info(f"Crawling page {page}, URL: {search_url}")
                        
                        # 페이지 접속
                        driver.get(search_url)
                        
                        # 페이지 로딩 대기
                        WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, ".api_txt_lines.total_tit"))
                        )
                        
                        # 랜덤 스크롤 (봇 감지 회피)
                        self._random_scroll(driver)
                        
                        # HTML 파싱
                        soup = BeautifulSoup(driver.page_source, 'html.parser')
                        
                        # 블로그 글 항목 추출
                        blog_items = soup.select(".api_txt_lines.total_tit")
                        
                        if not blog_items:
                            self.logger.warning(f"No blog items found on page {page}")
                            break
                        
                        # 블로그 글 정보 추출
                        for item in blog_items:
                            blog_card = item.find_parent(".total_wrap.api_ani_send")
                            
                            if not blog_card:
                                continue
                            
                            # 제목 및 URL
                            title_elem = blog_card.select_one(".api_txt_lines.total_tit")
                            title = title_elem.get_text(strip=True) if title_elem else ""
                            url = title_elem.get('href') if title_elem else ""
                            
                            # 요약 내용
                            desc_elem = blog_card.select_one(".api_txt_lines.dsc_txt")
                            description = desc_elem.get_text(strip=True) if desc_elem else ""
                            
                            # 블로그 이름
                            blog_name_elem = blog_card.select_one(".sub_txt.sub_name")
                            blog_name = blog_name_elem.get_text(strip=True) if blog_name_elem else ""
                            
                            # 날짜
                            date_elem = blog_card.select_one(".sub_time.sub_txt")
                            date = date_elem.get_text(strip=True) if date_elem else ""
                            
                            # 데이터 정제
                            clean_title = self.clean_text(title)
                            clean_desc = self.clean_text(description)
                            
                            # 유효 데이터만 추가
                            if clean_title and clean_desc and url:
                                blog_data = {
                                    'title': clean_title,
                                    'content': clean_desc,
                                    'url': url,
                                    'blog_name': blog_name,
                                    'published_date': date,
                                    'platform': 'naver_blog',
                                    'keyword': keyword,
                                    'crawled_at': time.strftime("%Y-%m-%d %H:%M:%S")
                                }
                                keyword_results.append(blog_data)
                        
                        # 랜덤 딜레이 (봇 감지 회피)
                        time.sleep(random.uniform(1.5, 3.0))
                        
                    except Exception as e:
                        self.logger.error(f"Error on page {page}: {str(e)}")
                
                # 결과 저장
                if keyword_results:
                    all_results.extend(keyword_results)
                    filename = self.generate_filename(keyword)
                    DataSaver.save_json(keyword_results, filename, self.save_dir)
                    self.logger.info(f"Saved {len(keyword_results)} results for keyword '{keyword}'")
                
        except Exception as e:
            self.logger.error(f"Crawler error: {str(e)}")
            
        finally:
            driver.quit()
            self.logger.info("Crawler finished")
            
        return all_results
    
    def _random_scroll(self, driver):
        """랜덤 스크롤 동작 (봇 감지 방지)"""
        total_height = driver.execute_script("return document.body.scrollHeight")
        for i in range(1, 4):
            scroll_height = total_height * (i / 4)
            driver.execute_script(f"window.scrollTo(0, {scroll_height});")
            time.sleep(random.uniform(0.5, 1.0)) 