import re
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import requests
from selenium import webdriver

class ContentExtractor:
    """웹 페이지에서 본문 내용을 추출하는 클래스"""
    
    def __init__(self, browser_type="chrome"):
        """
        ContentExtractor 초기화
        
        Args:
            browser_type: 사용할 브라우저 타입 ("chrome" 또는 "firefox")
        """
        self.browser_type = browser_type
        
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
        self.driver = None
        
    def init_driver(self):
        """웹드라이버 초기화"""
        try:
            if self.browser_type == "chrome":
                self.driver = webdriver.Chrome(options=self.options)
            elif self.browser_type == "firefox":
                self.driver = webdriver.Firefox(options=self.options)
            else:
                raise ValueError(f"지원되지 않는 브라우저 타입: {self.browser_type}")
            
            self.driver.set_page_load_timeout(30)
            return True
        except Exception as e:
            print(f"드라이버 초기화 실패: {str(e)}")
            return False

def extract_content(url, driver=None):
    """
    URL에서 내용을 추출하는 함수
    
    Args:
        url (str): 추출할 콘텐츠의 URL
        driver (WebDriver, optional): Selenium WebDriver 인스턴스
        
    Returns:
        str: 추출된 콘텐츠 텍스트 또는 None (추출 실패시)
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # 1. 블로그 URL 정규화 (모바일 URL을 데스크톱 URL로 변환)
    if 'm.blog.naver.com' in url:
        url = url.replace('m.blog.naver.com', 'blog.naver.com')
    
    # 2. 네이버 블로그 특수 처리
    if 'blog.naver.com' in url:
        return extract_naver_blog(url, driver, headers)
    
    # 3. 네이버 카페 특수 처리
    elif 'cafe.naver.com' in url:
        return extract_naver_cafe(url, driver, headers)
    
    # 4. 네이버 뉴스 특수 처리
    elif 'news.naver.com' in url:
        return extract_naver_news(url, driver, headers)
    
    # 5. 일반 페이지 처리
    return extract_general_content(url, driver, headers)

def extract_naver_blog(url, driver, headers):
    """네이버 블로그 콘텐츠 추출"""
    # 1. 정적 요청 시도
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # iframe URL 추출
            iframe = soup.find('iframe', id='mainFrame')
            if iframe:
                iframe_url = 'https://blog.naver.com' + iframe.get('src')
                response = requests.get(iframe_url, headers=headers)
                soup = BeautifulSoup(response.text, 'html.parser')
            
            # 네이버 블로그 포스트 ID 추출
            post_id_match = re.search(r'logNo=(\d+)', url)
            if post_id_match:
                post_id = post_id_match.group(1)
                blog_id_match = re.search(r'blog.naver.com/([^/]+)', url)
                if blog_id_match:
                    blog_id = blog_id_match.group(1)
                    
                    # 네이버 블로그 API로 직접 요청도 시도
                    api_url = f"https://blog.naver.com/PostView.naver?blogId={blog_id}&logNo={post_id}"
                    response = requests.get(api_url, headers=headers)
                    soup = BeautifulSoup(response.text, 'html.parser')
            
            # 다양한 클래스 선택자로 시도 (최신->구버전 순)
            selectors = [
                'div.se-main-container',  # 스마트에디터 ONE
                'div.__se_component_area',  # 구버전 에디터
                'div#viewTypeSelector',   # 구버전 에디터
                'div.post-view',
                'div.post_ct',
                'div.se_component_wrap'
            ]
            
            for selector in selectors:
                content = soup.select_one(selector)
                if content:
                    text = content.get_text(strip=True)
                    if _is_valid_content(text):
                        return text
    except Exception as e:
        print(f"Static extraction failed for Naver Blog: {str(e)}")
    
    # 2. Selenium 사용 시도 (정적 요청 실패 시)
    if driver:
        try:
            driver.get(url)
            # 페이지 로딩 대기
            time.sleep(2)
            
            # iframe이 있는지 확인
            frames = driver.find_elements(By.TAG_NAME, "iframe")
            main_frame = None
            for frame in frames:
                if frame.get_attribute('id') == 'mainFrame':
                    main_frame = frame
                    break
            
            # iframe으로 전환
            if main_frame:
                driver.switch_to.frame(main_frame)
                
                # 다양한 콘텐츠 영역 선택자 시도
                selectors = [
                    'div.se-main-container',  # 스마트에디터 ONE
                    'div.__se_component_area',  # 구버전 에디터
                    'div#viewTypeSelector',   # 구버전 에디터
                    'div.post-view',
                    'div.post_ct',
                    'div.se_component_wrap'
                ]
                
                for selector in selectors:
                    try:
                        # 명시적 대기 적용
                        WebDriverWait(driver, 5).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                        content = driver.find_element(By.CSS_SELECTOR, selector)
                        if content:
                            text = content.text
                            if _is_valid_content(text):
                                driver.switch_to.default_content()
                                return text
                    except:
                        continue
                
                driver.switch_to.default_content()
            
            # 기본 콘텐츠 추출 시도
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            for selector in selectors:
                content = soup.select_one(selector)
                if content:
                    text = content.get_text(strip=True)
                    if _is_valid_content(text):
                        return text
        except Exception as e:
            print(f"Dynamic extraction failed for Naver Blog: {str(e)}")
    
    return None

def extract_naver_cafe(url, driver, headers):
    """네이버 카페 콘텐츠 추출"""
    # 카페는 Selenium이 필수적
    if not driver:
        print("Selenium driver is required for Naver Cafe extraction")
        return None
    
    try:
        driver.get(url)
        time.sleep(2)
        
        # 로그인 팝업 닫기 시도
        try:
            close_button = driver.find_element(By.CSS_SELECTOR, "a.layer_close")
            close_button.click()
            time.sleep(1)
        except:
            pass
        
        # 메인 프레임으로 전환
        # 카페 게시물은 항상 iframe 내부에 있음
        try:
            # 카페 아티클 iframe 찾기
            cafe_main_frame = driver.find_element(By.ID, "cafe_main")
            driver.switch_to.frame(cafe_main_frame)
            
            # 다양한 콘텐츠 영역 선택자 시도
            selectors = [
                'div.se-main-container',  # 스마트에디터 ONE
                'div.article_container',
                'div.ContentRenderer',
                'div.article',
                'div#tbody'
            ]
            
            for selector in selectors:
                try:
                    content = driver.find_element(By.CSS_SELECTOR, selector)
                    if content:
                        text = content.text
                        if _is_valid_content(text):
                            driver.switch_to.default_content()
                            return text
                except:
                    continue
            
            driver.switch_to.default_content()
        except Exception as e:
            print(f"Failed to extract from cafe iframe: {e}")
            driver.switch_to.default_content()
        
        # 기본 콘텐츠 추출 시도
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        body = soup.find('body')
        if body:
            text = body.get_text(strip=True)
            if _is_valid_content(text):
                return text
    except Exception as e:
        print(f"Failed to extract Naver Cafe content: {str(e)}")
    
    return None

def extract_naver_news(url, driver, headers):
    """네이버 뉴스 콘텐츠 추출"""
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 다양한 뉴스 콘텐츠 ID 시도
            for id_name in ['dic_area', 'articleBodyContents', 'newsEndContents', 'articeBody']:
                content = soup.find('div', id=id_name)
                if content:
                    # 불필요한 요소 제거
                    for unnecessary in content.select('script, style, a.link_news, a.link_end'):
                        unnecessary.decompose()
                    
                    text = content.get_text(strip=True)
                    text = re.sub(r'\s+', ' ', text)  # 연속된 공백 제거
                    
                    if _is_valid_content(text):
                        return text
    except Exception as e:
        print(f"Static extraction failed for Naver News: {str(e)}")
    
    # Selenium 백업 추출
    if driver:
        try:
            driver.get(url)
            time.sleep(1)
            
            for id_name in ['dic_area', 'articleBodyContents', 'newsEndContents', 'articeBody']:
                try:
                    content = driver.find_element(By.ID, id_name)
                    if content:
                        text = content.text
                        if _is_valid_content(text):
                            return text
                except:
                    continue
        except Exception as e:
            print(f"Dynamic extraction failed for Naver News: {str(e)}")
    
    return None

def extract_general_content(url, driver, headers):
    """일반 웹페이지 콘텐츠 추출"""
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 일반적인 웹페이지 콘텐츠 영역 시도
            selectors = [
                'article',
                'div.content', 
                'div.article-content',
                'div.post-content',
                'main',
                'div.main-content'
            ]
            
            for selector in selectors:
                content = soup.select_one(selector)
                if content:
                    text = content.get_text(strip=True)
                    if _is_valid_content(text):
                        return text
            
            # 최후의 수단: body 전체 텍스트 사용
            body = soup.find('body')
            if body:
                text = body.get_text(strip=True)
                if _is_valid_content(text):
                    return text
    except Exception as e:
        print(f"Static extraction failed for general URL: {str(e)}")
    
    # Selenium 백업 추출
    if driver:
        try:
            driver.get(url)
            time.sleep(2)
            
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            for selector in ['article', 'div.content', 'main']:
                content = soup.select_one(selector)
                if content:
                    text = content.get_text(strip=True)
                    if _is_valid_content(text):
                        return text
            
            # 최후의 수단: body 전체 텍스트 사용
            body = soup.find('body')
            if body:
                text = body.get_text(strip=True)
                if _is_valid_content(text):
                    return text
        except Exception as e:
            print(f"Dynamic extraction failed for general URL: {str(e)}")
    
    return None

def _is_valid_content(text):
    """
    텍스트가 유효한 콘텐츠인지 확인 (한글 글자 수 50자 이상)
    
    Args:
        text (str): 확인할 텍스트
        
    Returns:
        bool: 유효한 콘텐츠인 경우 True, 그렇지 않으면 False
    """
    if not text:
        return False
        
    # 한글 글자 수 확인 (가-힣 범위의 문자)
    hangul_chars = re.findall(r"[가-힣]", text)
    return len(hangul_chars) >= 50 