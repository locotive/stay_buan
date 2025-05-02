import os
import re
import sys
import pytest
from bs4 import BeautifulSoup
from unittest.mock import MagicMock, patch

# 상위 디렉토리 추가 (utils 모듈 import를 위해)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.content_extractor import extract_content, _is_valid_content

# HTML 픽스처
@pytest.fixture
def naver_blog_html():
    return """
    <html>
    <body>
        <div class="se-main-container">
            여기는 네이버 블로그 본문입니다. 한글로 작성된 내용이 충분히 있어야 합니다.
            부안군은 전라북도의 서쪽 해안에 위치한 군입니다. 변산반도 국립공원이 있으며, 
            채석강, 격포 등 아름다운 해안 경관으로 유명합니다. 또한 부안은 젓갈과 같은 
            수산물 특산품이 유명하며 매년 다양한 축제가 열립니다. 부안군의 자연과 문화는 
            관광객들에게 독특한 경험을 제공합니다.
        </div>
    </body>
    </html>
    """

@pytest.fixture
def naver_news_html():
    return """
    <html>
    <body>
        <div id="dic_area">
            여기는 네이버 뉴스 본문입니다. 전라북도 부안군에서 새로운 관광 정책을 발표했습니다.
            부안군은 변산반도를 중심으로 생태관광을 활성화하고, 지역 특산품을 활용한 
            체험 프로그램을 확대할 계획이라고 밝혔습니다. 이번 정책은 코로나19 이후 
            침체된 지역 관광산업을 되살리기 위한 노력의 일환으로 보입니다.
            지역 주민들은 이번 정책에 대해 긍정적인 반응을 보이고 있습니다.
        </div>
    </body>
    </html>
    """

@pytest.fixture
def invalid_content_html():
    return """
    <html>
    <body>
        <div class="content">
            This is English content with very few Korean characters.
            한글 10자 미만
        </div>
    </body>
    </html>
    """

def test_is_valid_content():
    # 유효한 콘텐츠 (한글 50자 이상)
    valid_text = "안녕하세요. " * 10  # 한글 50자 이상
    assert _is_valid_content(valid_text) == True
    
    # 유효하지 않은 콘텐츠 (한글 50자 미만)
    invalid_text = "안녕하세요"  # 한글 5자
    assert _is_valid_content(invalid_text) == False
    
    # 영어만 있는 콘텐츠
    english_only = "Hello World! This is a test."
    assert _is_valid_content(english_only) == False
    
    # None 값
    assert _is_valid_content(None) == False
    
    # 빈 문자열
    assert _is_valid_content("") == False

def test_extract_content_requests(mocker, naver_blog_html, naver_news_html, invalid_content_html):
    # requests 응답 모킹
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = naver_blog_html
    
    # requests.get 모킹
    mocker.patch('requests.get', return_value=mock_response)
    
    # 네이버 블로그 URL에서 콘텐츠 추출 테스트
    url = "https://blog.naver.com/test_blog"
    content = extract_content(url)
    
    # 추출된 콘텐츠가 유효한지 확인
    assert content is not None
    assert '부안군' in content
    assert '변산반도' in content
    
    # 네이버 뉴스 URL 테스트
    mock_response.text = naver_news_html
    url = "https://news.naver.com/test_news"
    content = extract_content(url)
    
    # 추출된 콘텐츠가 유효한지 확인
    assert content is not None
    assert '부안군' in content
    assert '관광 정책' in content
    
    # 유효하지 않은 콘텐츠 테스트
    mock_response.text = invalid_content_html
    content = extract_content(url)
    
    # 한글 50자 미만이므로 None이 반환되어야 함
    assert content is None

def test_extract_content_selenium(mocker, naver_blog_html):
    # Selenium 드라이버 모킹
    mock_driver = MagicMock()
    mock_driver.page_source = naver_blog_html
    mock_driver.find_elements.return_value = []  # iframe이 없다고 가정
    
    # requests.get이 실패하도록 모킹
    mocker.patch('requests.get', side_effect=Exception("Connection error"))
    
    # 네이버 블로그 URL에서 Selenium을 사용하여 콘텐츠 추출 테스트
    url = "https://blog.naver.com/test_blog"
    content = extract_content(url, mock_driver)
    
    # 추출된 콘텐츠가 유효한지 확인
    assert content is not None
    assert '부안군' in content
    assert '변산반도' in content
    
    # iframe이 있는 경우 테스트
    mock_frame = MagicMock()
    mock_driver.find_elements.return_value = [mock_frame]
    mock_driver.page_source = naver_blog_html
    
    # switch_to.frame 호출 후 반환되는 페이지 설정
    def switch_frame(frame):
        # driver.page_source가 iframe 내부 콘텐츠를 반환한다고 가정
        pass
    
    mock_driver.switch_to.frame = switch_frame
    
    content = extract_content(url, mock_driver)
    
    # 추출된 콘텐츠가 유효한지 확인
    assert content is not None
    assert '부안군' in content
    
    # 예외 처리 테스트
    mock_driver.get.side_effect = Exception("Selenium error")
    content = extract_content(url, mock_driver)
    assert content is None 