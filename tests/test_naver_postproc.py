import unittest
import sys
import os
import datetime
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from crawlers.naver_api_crawler import NaverSearchAPICrawler, contains_all, DuplicateDocError

class TestNaverPostprocessing(unittest.TestCase):
    def setUp(self):
        self.keywords = [{'text': '부안'}, {'text': '맛집'}]
        self.crawler = NaverSearchAPICrawler(self.keywords, max_pages=1)
        
        # 테스트용 아이템 생성
        self.now = datetime.now()
        
        # 1. 키워드가 일부만 포함된 문서
        self.partial_kw_item = {
            'title': '부안 여행',
            'content': '부안의 여행 명소를 소개합니다.',
            'url': 'https://example.com/partial',
            'blog_name': '블로그B',
            'published_date': self.now.strftime('%Y%m%d'),
            'date_obj': self.now.isoformat(),
            'platform': 'naver_blog_api',
            'keyword': '부안',
            'original_keywords': '부안,맛집',
            'sentiment': None,
            'crawled_at': self.now.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 2. 유효한 문서 (모든 키워드 포함)
        self.valid_item = {
            'title': '부안 맛집 탐방',
            'content': '부안에서 맛집을 찾아 탐방했습니다.',
            'url': 'https://example.com/valid',
            'blog_name': '블로그C',
            'published_date': self.now.strftime('%Y%m%d'),
            'date_obj': self.now.isoformat(),
            'platform': 'naver_blog_api',
            'keyword': '부안 맛집',
            'original_keywords': '부안,맛집',
            'sentiment': None,
            'crawled_at': self.now.strftime('%Y-%m-%d %H:%M:%S')
        }
        
    def test_contains_all(self):
        """키워드 포함 검사 함수 테스트"""
        # 모든 키워드가 포함된 경우
        text = "부안에 있는 맛집을 소개합니다."
        keywords = ["부안", "맛집"]
        self.assertTrue(contains_all(text, keywords))
        
        # 일부 키워드만 포함된 경우
        text = "부안 여행 코스"
        keywords = ["부안", "맛집"]
        self.assertFalse(contains_all(text, keywords))
        
        # 대소문자 구분 없이 작동하는지 테스트
        text = "부안 BEST 맛집 추천"
        keywords = ["부안", "맛집"]
        self.assertTrue(contains_all(text, keywords))
        
    def test_postprocess_keyword_filter(self):
        """키워드 필터링 테스트"""
        items = [self.partial_kw_item, self.valid_item]
        result = self.crawler._postprocess(items, ["부안", "맛집"])
        
        # 키워드가 일부만 있는 문서는 필터링되어야 함
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['url'], 'https://example.com/valid')
        
    def test_doc_id_generation(self):
        """문서 ID 생성 테스트"""
        items = [self.valid_item]
        result = self.crawler._postprocess(items, ["부안", "맛집"])
        
        # doc_id가 생성되어야 함
        self.assertTrue('doc_id' in result[0])
        # sha256 해시는 64자리 16진수 문자열
        self.assertEqual(len(result[0]['doc_id']), 64)
        
    def test_duplicate_detection(self):
        """중복 문서 감지 테스트"""
        # 동일한 URL과 제목으로 두 개의 문서 생성
        duplicate_item = self.valid_item.copy()
        duplicate_item['content'] = "내용이 다르지만 URL과 제목은 같습니다."
        
        items = [self.valid_item, duplicate_item]
        
        # 첫 번째 항목은 정상 처리
        result = self.crawler._postprocess([self.valid_item], ["부안", "맛집"])
        self.assertEqual(len(result), 1)
        
        # 두 번째 중복 항목은 예외 발생해야 함
        with self.assertRaises(DuplicateDocError):
            self.crawler._postprocess([duplicate_item], ["부안", "맛집"])
            
    def test_complete_pipeline(self):
        """전체 파이프라인 테스트"""
        # 모든 테스트 아이템
        all_items = [self.partial_kw_item, self.valid_item]
        
        # 후처리 후에는 valid_item만 남아야 함
        result = self.crawler._postprocess(all_items, ["부안", "맛집"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['url'], 'https://example.com/valid')
        self.assertTrue('doc_id' in result[0])
        
if __name__ == '__main__':
    unittest.main() 